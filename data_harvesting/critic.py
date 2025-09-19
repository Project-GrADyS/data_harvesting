import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MultiAgentMLP

def get_activation_class(name):
    if name == "Tanh":
        return torch.nn.Tanh
    elif name == "ReLU":
        return torch.nn.ReLU
    elif name == "LeakyReLU":
        return torch.nn.LeakyReLU
    else:
        raise ValueError(f"Unknown activation function: {name}")

def create_critic(env, device, config):
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
