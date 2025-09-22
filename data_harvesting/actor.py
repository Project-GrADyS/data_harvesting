import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase
from torchrl.modules import (
    ProbabilisticActor,
    MultiAgentMLP,
    AdditiveGaussianModule,
    NormalParamWrapper,
)
from torchrl.modules.distributions import TanhNormal


def get_activation_class(name):
    if name == "Tanh":
        return torch.nn.Tanh
    elif name == "ReLU":
        return torch.nn.ReLU
    elif name == "LeakyReLU":
        return torch.nn.LeakyReLU
    else:
        raise ValueError(f"Unknown activation function: {name}")


def create_actor(
        env: EnvBase,
        device: torch.device,
        config: dict) -> TensorDictModule:
    """
    Creates a multi-agent DDPG actor.
    :param env: The environment.
    :param device: The device to run the actor on.
    :param config: Configuration dictionary.
    :return: The actor as a TensorDictModule.
    """
    activation_class = get_activation_class(config["actor"]["activation_function"])
    policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[("agents", "observation")].shape[-1],
        n_agent_outputs=env.full_action_spec[("agents", "action")].shape[-1],
        n_agents=config["environment"]["num_drones"],
        centralised=config["actor"]["centralized"],
        share_params=config["actor"]["share_parameters"],
        device=device,
        depth=config["actor"]["network_depth"],
        num_cells=config["actor"]["network_width"],
        activation_class=activation_class
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "param")],
    )

    from torchrl.modules import TanhDelta

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec[("agents", "action")],
        in_keys=[("agents", "param")],
        out_keys=[("agents", "action")],
        distribution_class=TanhDelta,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched["agents", "action"].space.low,
            "high": env.full_action_spec_unbatched["agents", "action"].space.high,
        },
        return_log_prob=False,
    )
    return policy


def create_exploratory_actor(
        actor: TensorDictModule,
        device: torch.device,
        config: dict) -> tuple[TensorDictModule, AdditiveGaussianModule]:
    """
    Creates an exploratory actor by adding Gaussian noise to the actions of the given actor.
    :param actor: The base actor.
    :param device: The device to run the exploratory actor on.
    :param config: Configuration dictionary.
    :return: The exploratory actor and the exploration noise module.
    """
    exploration_noise = AdditiveGaussianModule(
        spec=actor.spec,
        annealing_num_steps=config["training"]["exploration_annealing_steps"],
        action_key=("agents", "action"),
        sigma_init=config["training"]["exploration_sigma_init"],
        sigma_end=config["training"]["exploration_sigma_end"],
        device=device
    )

    return TensorDictSequential(
        actor,
        exploration_noise
    ), exploration_noise


def create_ppo_actor(
        env: EnvBase,
        device: torch.device,
        config: dict) -> TensorDictModule:
    """
    Creates a multi-agent PPO actor with a squashed Gaussian policy.
    Outputs actions in the env's action bounds and stores sample_log_prob for PPO.
    """
    activation_class = get_activation_class(config["actor"]["activation_function"])
    action_dim = env.full_action_spec[("agents", "action")].shape[-1]

    # Policy backbone produces concatenated normal params (mean, std) per action dim
    policy_backbone = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[("agents", "observation")].shape[-1],
        n_agent_outputs=action_dim * 2,
        n_agents=config["environment"]["num_drones"],
        centralised=config["actor"]["centralized"],
        share_params=config["actor"]["share_parameters"],
        device=device,
        depth=config["actor"]["network_depth"],
        num_cells=config["actor"]["network_width"],
        activation_class=activation_class,
    )

    base_module = TensorDictModule(
        policy_backbone,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "param")],
    )

    # Wrap to split into (loc, scale) and ensure positive std
    param_wrapped = NormalParamWrapper(
        base_module,
        in_keys=[("agents", "param")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    # Squashed Gaussian policy with log-prob output
    policy = ProbabilisticActor(
        module=param_wrapped,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        spec=env.full_action_spec[("agents", "action")],
        distribution_class=TanhNormal,
        # TanhNormal will be created from loc/scale; env spec handles bounds
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    return policy