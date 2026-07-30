from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from flex_marl import (
    CentralizedOutput,
    FlatFieldConfig,
    MultiAgentEncoderConfig,
    MultiAgentEncoderModule,
    MultiAgentMode,
    SequentialFieldConfig,
    SequentialFieldOptions,
)
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.envs import EnvBase
from torchrl.modules import EGreedyModule

from kaz_training.config import get_activation


class _StraightThroughOneHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probabilities):
        return torch.nn.functional.one_hot(
            probabilities.argmax(dim=-1),
            num_classes=probabilities.shape[-1],
        ).to(probabilities.dtype)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class _ActorNetwork(nn.Module):
    def __init__(self, encoder: MultiAgentEncoderModule) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, entities, entity_mask, role, agent_mask):
        logits = self.encoder(
            {
                "entities": entities,
                "entity_mask": entity_mask,
                "role": role,
                "agent_mask": agent_mask,
            }
        )
        probabilities = logits.softmax(dim=-1)
        action = _StraightThroughOneHot.apply(probabilities)
        return action * agent_mask.unsqueeze(-1).to(action.dtype)


class _CriticNetwork(nn.Module):
    def __init__(self, encoder: MultiAgentEncoderModule) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, entities, entity_mask, role, action, agent_mask):
        return self.encoder(
            {
                "entities": entities,
                "entity_mask": entity_mask,
                "role": role,
                "action": action,
                "agent_mask": agent_mask,
            }
        )


class _MaskInactiveActions(nn.Module):
    def forward(self, action, agent_mask):
        return action * agent_mask.unsqueeze(-1).to(action.dtype)


def _sequential_options(config: Mapping[str, Any], *, encode_agent_identity: bool):
    encoder_config = config["encoder"]
    return SequentialFieldOptions(
        num_heads=int(encoder_config["num_heads"]),
        ff_dim=int(encoder_config["ff_dim"]),
        depth=int(encoder_config["depth"]),
        dropout=float(encoder_config["dropout"]),
        encode_agent_identity=encode_agent_identity,
    )


def _common_encoder_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    encoder_config = config["encoder"]
    return {
        "mix_layer_depth": int(encoder_config["mix_layer_depth"]),
        "mix_layer_num_cells": int(encoder_config["mix_layer_num_cells"]),
        "mix_activation_class": get_activation(str(encoder_config["activation_function"])),
    }


def create_actor(env: EnvBase, config: Mapping[str, Any], device: torch.device):
    num_agents = env.full_action_spec[("agents", "action")].shape[-2]
    action_dim = env.full_action_spec[("agents", "action")].shape[-1]
    options = _sequential_options(config, encode_agent_identity=False)
    encoder = MultiAgentEncoderModule(
        MultiAgentEncoderConfig(
            fields=(
                SequentialFieldConfig(
                    key="entities",
                    mask_key="entity_mask",
                    input_size=11,
                    output_size=int(config["actor"]["entity_embed_dim"]),
                    sequential_options=options,
                ),
                FlatFieldConfig(
                    key="role",
                    input_size=2,
                    output_size=int(config["actor"]["role_embed_dim"]),
                    depth=1,
                    hidden_layer_size=int(config["actor"]["role_embed_dim"]),
                    activation_class=get_activation(str(config["encoder"]["activation_function"])),
                    sequential_options=options,
                ),
            ),
            num_agents=num_agents,
            mode=MultiAgentMode.SHARED,
            agent_mask_key="agent_mask",
            output_dim=action_dim,
            centralized_output=CentralizedOutput.GLOBAL,
            **_common_encoder_kwargs(config),
        ),
        device=device,
        run_pre_forward_checks=not bool(config["encoder"].get("compile", False)),
    )
    if bool(config["encoder"].get("compile", False)):
        encoder = torch.compile(encoder)
    return TensorDictModule(
        _ActorNetwork(encoder),
        in_keys=[
            ("agents", "observation", "entities"),
            ("agents", "observation", "entity_mask"),
            ("agents", "observation", "role"),
            ("agents", "mask"),
        ],
        out_keys=[("agents", "action")],
    )


def create_exploratory_actor(actor, env: EnvBase, config, device):
    explorer = EGreedyModule(
        spec=env.full_action_spec[("agents", "action")],
        eps_init=float(config["training"]["exploration_epsilon_init"]),
        eps_end=float(config["training"]["exploration_epsilon_end"]),
        annealing_num_steps=int(config["training"]["exploration_annealing_steps"]),
        action_key=("agents", "action"),
        device=device,
    )
    mask_module = TensorDictModule(
        _MaskInactiveActions(),
        in_keys=[("agents", "action"), ("agents", "mask")],
        out_keys=[("agents", "action")],
    )
    return TensorDictSequential(actor, explorer, mask_module), explorer


def create_critic(env: EnvBase, config: Mapping[str, Any], device: torch.device):
    num_agents = env.full_action_spec[("agents", "action")].shape[-2]
    action_dim = env.full_action_spec[("agents", "action")].shape[-1]
    options = _sequential_options(config, encode_agent_identity=True)
    activation = get_activation(str(config["encoder"]["activation_function"]))
    encoder = MultiAgentEncoderModule(
        MultiAgentEncoderConfig(
            fields=(
                SequentialFieldConfig(
                    key="entities",
                    mask_key="entity_mask",
                    input_size=11,
                    output_size=int(config["critic"]["entity_embed_dim"]),
                    sequential_options=options,
                ),
                FlatFieldConfig(
                    key="role",
                    input_size=2,
                    output_size=int(config["critic"]["role_embed_dim"]),
                    depth=1,
                    hidden_layer_size=int(config["critic"]["role_embed_dim"]),
                    activation_class=activation,
                    sequential_options=options,
                ),
                FlatFieldConfig(
                    key="action",
                    input_size=action_dim,
                    output_size=int(config["critic"]["action_embed_dim"]),
                    depth=1,
                    hidden_layer_size=int(config["critic"]["action_embed_dim"]),
                    activation_class=activation,
                    sequential_options=options,
                ),
            ),
            num_agents=num_agents,
            mode=MultiAgentMode.CENTRALIZED,
            agent_mask_key="agent_mask",
            output_dim=1,
            centralized_output=CentralizedOutput.GLOBAL,
            **_common_encoder_kwargs(config),
        ),
        device=device,
        run_pre_forward_checks=not bool(config["encoder"].get("compile", False)),
    )
    if bool(config["encoder"].get("compile", False)):
        encoder = torch.compile(encoder)
    return TensorDictModule(
        _CriticNetwork(encoder),
        in_keys=[
            ("agents", "observation", "entities"),
            ("agents", "observation", "entity_mask"),
            ("agents", "observation", "role"),
            ("agents", "action"),
            ("agents", "mask"),
        ],
        out_keys=["state_action_value"],
    )
