import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MultiAgentMLP

from data_harvesting.utils import get_activation_class

def create_critic(env, device, config):
    """Creates a multi-agent critic Q(s, a) for MADDPG."""
    if config["environment"]["sequential_obs"]:
        raise NotImplementedError("MLP Critic not implemented for sequential observations.")

    if config["flex_encoder"]["enabled"]:
        obs_key = ("agents", "encoded_obs")
        obs_size = config["flex_encoder"]["output_dim"]
    else:
        obs_key = ("agents", "observation")
        obs_size = env.observation_spec[("agents", "observation")].shape[-1]

    cat_module = TensorDictModule(
        lambda obs, action: torch.cat([obs, action], dim=-1),
        in_keys=[obs_key, ("agents", "action")],
        out_keys=[("agents", "obs_action")],
    )

    critic_params = config["critic"]
    activation_class = get_activation_class(critic_params["activation_function"])

    critic_module = TensorDictModule(
        module=MultiAgentMLP(
            n_agent_inputs=obs_size + env.full_action_spec["agents", "action"].shape[-1],
            n_agent_outputs=1,
            n_agents=config["environment"]["num_drones"],
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

def create_value_net(env, device, config):
    """Creates a multi-agent value network V(s) for PPO/MAPPO."""
    critic_params = config["critic"]
    activation_class = get_activation_class(critic_params["activation_function"])

    value_module = TensorDictModule(
        module=MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=1,
            n_agents=config["environment"]["num_drones"],
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
