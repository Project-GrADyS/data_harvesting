"""Shared DQN policy construction for training, checkpoints, and leagues."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from flex_marl import (
    FlatFieldConfig,
    MultiAgentEncoderConfig,
    MultiAgentEncoderModule,
    MultiAgentMode,
    SequentialFieldConfig,
    SequentialFieldOptions,
)
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.modules import MultiAgentMLP, QValueModule

from .environment import N_TYPES
from .environment.torchrl import (
    ACTION_KEY,
    ACTION_VALUE_KEY,
    CHOSEN_VALUE_KEY,
    ENCODER_KEY,
    GROUP,
    HAND_HISTORY_MASK_KEY,
    MASK_KEY,
    OBS_KEY,
    OPPONENT_TABLEAUS_MASK_KEY,
    PLAYER_MASK_KEY,
    flat_observation_dim,
    make_env,
    make_observation_flattener,
    resolve_player_counts,
)

CHECKPOINT_SCHEMA_VERSION = 1


class _EncoderRouter(nn.Module):
    def __init__(self, encoder: MultiAgentEncoderModule) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        prepared = dict(inputs)
        prepared["agent_mask"] = prepared["agent_mask"].squeeze(-1).bool()
        prepared["hand_history_mask"] = prepared["hand_history_mask"].bool()
        prepared["opponent_tableaus_mask"] = prepared[
            "opponent_tableaus_mask"
        ].bool()
        return self.encoder(prepared)


def _activation(name: str) -> type[nn.Module]:
    activations = {"Tanh": nn.Tanh, "ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU}
    try:
        return activations[name]
    except KeyError as error:
        raise ValueError(f"Unsupported activation function: {name}") from error


def make_qvalue_selector() -> QValueModule:
    return QValueModule(
        action_space="categorical",
        action_value_key=ACTION_VALUE_KEY,
        action_mask_key=MASK_KEY,
        out_keys=[ACTION_KEY, ACTION_VALUE_KEY, CHOSEN_VALUE_KEY],
    )


def _make_flex_encoder(
    environment,
    config: Mapping[str, Any],
    *,
    num_players: int,
    device: torch.device,
) -> TensorDictModule:
    encoder_config = config["model"]["encoder"]
    sequential = encoder_config["sequential"]
    flat = encoder_config["flat"]
    options = SequentialFieldOptions(
        num_heads=int(sequential["num_heads"]),
        ff_dim=int(sequential["ff_dim"]),
        depth=int(sequential["depth"]),
        dropout=float(sequential["dropout"]),
        encode_agent_identity=bool(sequential["encode_agent_identity"]),
    )
    fields = (
        SequentialFieldConfig(
            key="hand_history",
            mask_key="hand_history_mask",
            input_size=environment.observation_spec[
                (GROUP, "observation", "hand_history")
            ].shape[-1],
            output_size=int(sequential["embed_dim"]),
            sequential_options=options,
        ),
        SequentialFieldConfig(
            key="opponent_tableaus",
            mask_key="opponent_tableaus_mask",
            input_size=environment.observation_spec[
                (GROUP, "observation", "opponent_tableaus")
            ].shape[-1],
            output_size=int(sequential["embed_dim"]),
            sequential_options=options,
        ),
        *(
            FlatFieldConfig(
                key=key,
                input_size=environment.observation_spec[
                    (GROUP, "observation", key)
                ].shape[-1],
                output_size=int(flat["embed_dim"]),
                depth=int(flat["depth"]),
                hidden_layer_size=int(flat["num_cells"]),
                activation_class=_activation(str(flat["activation"])),
            )
            for key in (
                "current_hand",
                "own_tableau",
                "cards_played",
                "game_scalars",
            )
        ),
    )
    encoder = MultiAgentEncoderModule(
        MultiAgentEncoderConfig(
            fields=fields,
            num_agents=num_players,
            mode=MultiAgentMode.SHARED,
            agent_mask_key="agent_mask",
            output_dim=int(encoder_config["output_dim"]),
            mix_layer_depth=int(encoder_config["mix_depth"]),
            mix_layer_num_cells=int(encoder_config["mix_cells"]),
            mix_activation_class=_activation(
                str(encoder_config["mix_activation"])
            ),
        ),
        device=device,
        run_pre_forward_checks=not bool(encoder_config.get("compile", False)),
    )
    if bool(encoder_config.get("compile", False)):
        encoder = torch.compile(encoder)

    return TensorDictModule(
        _EncoderRouter(encoder),
        in_keys={
            "agent_mask": PLAYER_MASK_KEY,
            "hand_history": (GROUP, "observation", "hand_history"),
            "hand_history_mask": HAND_HISTORY_MASK_KEY,
            "opponent_tableaus": (
                GROUP,
                "observation",
                "opponent_tableaus",
            ),
            "opponent_tableaus_mask": OPPONENT_TABLEAUS_MASK_KEY,
            "current_hand": (GROUP, "observation", "current_hand"),
            "own_tableau": (GROUP, "observation", "own_tableau"),
            "cards_played": (GROUP, "observation", "cards_played"),
            "game_scalars": (GROUP, "observation", "game_scalars"),
        },
        out_keys=[ENCODER_KEY],
        out_to_in_map=True,
    )


def build_q_policy(
    environment,
    config: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> TensorDictSequential:
    """Build the masked shared-parameter Q policy from project configuration."""

    device = torch.device(device)
    _, _, _, num_players = resolve_player_counts(config["environment"])
    model_config = config["model"]
    if bool(model_config["use_encoder"]):
        encoder = _make_flex_encoder(
            environment, config, num_players=num_players, device=device
        )
        q_input_key = ENCODER_KEY
        q_input_size = int(model_config["encoder"]["output_dim"])
        prefix: list[nn.Module] = [encoder]
    else:
        q_input_key = OBS_KEY
        q_input_size = flat_observation_dim(environment)
        prefix = [make_observation_flattener()]

    q_network = MultiAgentMLP(
        n_agent_inputs=q_input_size,
        n_agent_outputs=N_TYPES,
        n_agents=num_players,
        centralised=False,
        share_params=True,
        depth=int(model_config["q_head"]["depth"]),
        num_cells=int(model_config["q_head"]["num_cells"]),
        activation_class=_activation(str(model_config["q_head"]["activation"])),
        device=device,
    )
    q_module = TensorDictModule(
        q_network,
        in_keys=[q_input_key],
        out_keys=[ACTION_VALUE_KEY],
    )
    return TensorDictSequential(*prefix, q_module, make_qvalue_selector())


def checkpoint_payload(
    policy: nn.Module, config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state_dict": policy.state_dict(),
        "model_config": dict(config["model"]),
        "environment_config": dict(config["environment"]),
    }


def load_checkpoint_policy(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Strictly load one integrated checkpoint and reconstruct its policy."""

    payload = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Sushi Go checkpoint schema: {payload.get('schema_version')}"
        )
    config = {
        "model": payload["model_config"],
        "environment": payload["environment_config"],
    }
    environment = make_env(config, device=device)
    try:
        policy = build_q_policy(environment, config, device=device)
        policy(environment.reset())
        policy.load_state_dict(payload["state_dict"], strict=True)
        policy.eval()
    finally:
        environment.close()
    return policy, config
