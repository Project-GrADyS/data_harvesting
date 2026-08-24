import torch
import torch.nn as nn
from typing import Dict, Any
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MultiAgentMLP
from torchrl.envs import EnvBase

from data_harvesting.utils import get_activation_class
from data_harvesting.encoding import MaskedMultiAgentMLP, make_flex_encoder_module
from data_harvesting.environment import requires_masking

def create_mlp_critic(env: EnvBase, config: Dict[str, Any], device: torch.device) -> TensorDictModule:
    """Creates a multi-agent critic Q(s, a) for MADDPG."""
    cat_module = TensorDictModule(
        lambda obs, action: torch.cat([obs, action], dim=-1),
        in_keys=[("agents", "observation", "flat"), ("agents", "action")],
        out_keys=[("agents", "obs_action")],
    )

    critic_params = config["critic"]
    activation_class = get_activation_class(critic_params["activation_function"])

    critic_module = TensorDictModule(
        module=MaskedMultiAgentMLP(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec["agents", "observation", "flat"].shape[-1]
                               + env.full_action_spec["agents", "action"].shape[-1],
                n_agent_outputs=1,
                n_agents=config["environment"]["max_num_agents"],
                centralised=critic_params["centralized"],
                share_params=critic_params["share_parameters"],
                device=device,
                depth=critic_params["network_depth"],
                num_cells=critic_params["network_width"],
                activation_class=activation_class,
            )
        ),
        in_keys=[("agents", "obs_action"), ("agents", "mask")],
        out_keys=[("agents", "state_action_value")],
    )

    critic = TensorDictSequential(cat_module, critic_module)
    return critic

def create_flex_critic(env: EnvBase, config: Dict[str, Any], device: torch.device) -> TensorDictModule:
    return make_flex_encoder_module(
        env=env,
        config=config,
        network_config=config["critic"],
        output_dim=1,
        output_key=("agents", "state_action_value"),
        include_action=True,
        encode_agent_identity=bool(config["flex_encoder"]["sequential_heads"]["critic_agent_embedding"]),
        device=device,
    )

def create_critic(env, device, config):
    return create_flex_critic(env, config, device) if config["flex_encoder"]["enabled"] else create_mlp_critic(env, config, device)

def create_ppo_value_net(env, device, config):
    """Creates a multi-agent value network V(s) for PPO/MAPPO."""
    if config["flex_encoder"]["enabled"]:
        raise ValueError("MAPPO requires flex_encoder.enabled to be false.")
    if requires_masking(config):
        raise NotImplementedError("PPO Value Network does not support environments that require masking.")

    critic_params = config["critic"]
    activation_class = get_activation_class(critic_params["activation_function"])

    value_module = TensorDictModule(
        module=MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation", "flat"].shape[-1],
            n_agent_outputs=1,
            n_agents=config["environment"]["max_num_agents"],
            centralised=critic_params["centralized"],
            share_params=critic_params["share_parameters"],
            device=device,
            depth=critic_params["network_depth"],
            num_cells=critic_params["network_width"],
            activation_class=activation_class,
        ),
        in_keys=[("agents", "observation", "flat")],
        out_keys=[("agents", "state_value")],
    )
    return value_module
