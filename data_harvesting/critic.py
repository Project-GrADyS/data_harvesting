import torch
import torch.nn as nn
from typing import Dict, Any
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MultiAgentMLP
from torchrl.envs import EnvBase

from data_harvesting.utils import get_activation_class
from data_harvesting.encoder import (
    MultiAgentFlexModule,
    SequentialEncoderConfig,
    FlatEncoderConfig,
)

def create_mlp_critic(env: EnvBase, config: Dict[str, Any], device: torch.device) -> TensorDictModule:
    """Creates a multi-agent critic Q(s, a) for MADDPG."""
    if config["environment"]["sequential_obs"]:
        raise NotImplementedError("MLP Critic not implemented for sequential observations.")
    
    if config["environment"]["min_num_drones"] != config["environment"]["max_num_drones"]:
        raise NotImplementedError("MLP Critic not implemented for variable number of drones.")

    cat_module = TensorDictModule(
        lambda obs, action: torch.cat([obs, action], dim=-1),
        in_keys=[("agents", "observation"), ("agents", "action")],
        out_keys=[("agents", "obs_action")],
    )

    critic_params = config["critic"]
    activation_class = get_activation_class(critic_params["activation_function"])

    critic_module = TensorDictModule(
        module=MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1]
                           + env.full_action_spec["agents", "action"].shape[-1],
            n_agent_outputs=1,
            n_agents=config["environment"]["max_num_drones"],
            centralised=critic_params["centralized"],
            share_params=critic_params["share_parameters"],
            device=device,
            depth=critic_params["network_depth"],
            num_cells=critic_params["network_width"],
            activation_class=activation_class,
        ),
        in_keys=[("agents", "obs_action")],
        out_keys=[("agents", "state_action_value")],
    )

    critic = TensorDictSequential(cat_module, critic_module)
    return critic

def create_flex_critic(env: EnvBase, config: Dict[str, Any], device: torch.device) -> TensorDictModule:
    flex_cfg = config["flex_encoder"]
    seq_heads_cfg = flex_cfg["sequential_heads"]
    flat_heads_cfg = flex_cfg["flat_heads"]

    sequential_configs = []
    flat_configs = []
    in_keys = {}

    env_is_sequential = config["environment"]["sequential_obs"]
    
    if env_is_sequential:
        # Configuration for the drones part of the observation
        sequential_configs.append(
            SequentialEncoderConfig(
                key="drones",
                input_size=env.observation_spec[("agents", "observation","drones")].shape[-1],
                embed_dim=seq_heads_cfg["embed_dim"],
                head_dim=seq_heads_cfg["head_dim"],
                num_heads=seq_heads_cfg["num_heads"],
                ff_dim=seq_heads_cfg["ff_dim"],
                depth=seq_heads_cfg["depth"],
                dropout=seq_heads_cfg["dropout"],
                max_num_agents=config["environment"]["max_num_drones"],
                agentic_encoding=seq_heads_cfg["critic_agent_embedding"]
            )
        )
        in_keys["drones"] = ("agents", "observation", "drones")
        # Sequential config for the sensors part of the observation
        sequential_configs.append(
            SequentialEncoderConfig(
                key="sensors",
                input_size=env.observation_spec[("agents", "observation","sensors")].shape[-1],
                embed_dim=seq_heads_cfg["embed_dim"],
                head_dim=seq_heads_cfg["head_dim"],
                num_heads=seq_heads_cfg["num_heads"],
                ff_dim=seq_heads_cfg["ff_dim"],
                depth=seq_heads_cfg["depth"],
                dropout=seq_heads_cfg["dropout"],
                max_num_agents=config["environment"]["max_num_drones"],
                agentic_encoding=seq_heads_cfg["critic_agent_embedding"]
            )
        )
        in_keys["sensors"] = ("agents", "observation", "sensors")
        if config["environment"]["id_on_state"]:
            # Flat config for the agent_id part of the observation
            flat_configs.append(
                FlatEncoderConfig(
                    key="agent_id",
                    input_size=env.observation_spec[("agents", "observation","agent_id")].shape[-1],
                    embed_dim=flat_heads_cfg["embed_dim"],
                    depth=flat_heads_cfg["depth"],
                    num_cells=flat_heads_cfg["num_cells"],
                    activation_class=get_activation_class(flat_heads_cfg["activation_function"])
                )
            )
            in_keys["agent_id"] = ("agents", "observation", "agent_id")
    else:
        # Flat config for the entire observation when not sequential
        flat_configs.append(
            FlatEncoderConfig(
                key="observation",
                input_size=env.observation_spec[("agents", "observation")].shape[-1],
                embed_dim=flat_heads_cfg["embed_dim"],
                depth=flat_heads_cfg["depth"],
                num_cells=flat_heads_cfg["num_cells"],
                activation_class=get_activation_class(flat_heads_cfg["activation_function"])
            )
        )
        in_keys["observation"] = ("agents", "observation")

    # Add flat config for the action space
    flat_configs.append(
        FlatEncoderConfig(
            key="action",
            input_size=env.full_action_spec[("agents", "action")].shape[-1],
            embed_dim=flat_heads_cfg["embed_dim"],
            depth=flat_heads_cfg["depth"],
            num_cells=flat_heads_cfg["num_cells"],
            activation_class=get_activation_class(flat_heads_cfg["activation_function"])
        )
    )
    in_keys["action"] = ("agents", "action")

    encoder = MultiAgentFlexModule(
        sequential_configs=sequential_configs, 
        flat_configs=flat_configs,
        mix_layer_depth=flex_cfg["mix_layer_depth"],
        mix_layer_num_cells=flex_cfg["mix_layer_num_cells"],
        mix_activation_class=get_activation_class(flex_cfg["mix_activation_function"]),
        output_dim=1,
        n_agents=config["environment"]["max_num_drones"],
        centralized=config["critic"]["centralized"],
        share_params=config["critic"]["share_parameters"],
        device=device
    )

    critic_module = TensorDictModule(
        encoder,
        in_keys=in_keys,
        out_keys=[("agents", "state_action_value")],
        out_to_in_map=True
    )
    return critic_module

def create_critic(env, device, config):
    return create_flex_critic(env, config, device) if config["flex_encoder"]["enabled"] else create_mlp_critic(env, config, device)

def create_ppo_value_net(env, device, config):
    """Creates a multi-agent value network V(s) for PPO/MAPPO."""
    if config["environment"]["max_num_drones"] != config["environment"]["min_num_drones"]:
        raise NotImplementedError("PPO Value Network not implemented for variable number of drones.")

    critic_params = config["critic"]
    activation_class = get_activation_class(critic_params["activation_function"])

    value_module = TensorDictModule(
        module=MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=1,
            n_agents=config["environment"]["max_num_drones"],
            centralised=critic_params["centralized"],
            share_params=critic_params["share_parameters"],
            device=device,
            depth=critic_params["network_depth"],
            num_cells=critic_params["network_width"],
            activation_class=activation_class,
        ),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )
    return value_module
