import torch
from typing import Dict, Any
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase
from torchrl.modules import (
    ProbabilisticActor,
    MultiAgentMLP,
    AdditiveGaussianModule,
    NormalParamExtractor,
)
from torchrl.modules.distributions import TanhNormal
from data_harvesting.encoder import (
    create_flex_encoder,
)
from data_harvesting.utils import get_activation_class


def create_policy_module(env: EnvBase, config: Dict[str, Any], device: torch.device) -> TensorDictModule:
    """
    Creates the policy module for the actor, with optional flex encoder. If flex encoder
    is enabled, the policy module will use encoded observations as input. Otherwise, it will
    use raw observations.
    """

    if config["environment"]["sequential_obs"]:
        raise NotImplementedError("MLP Actor not implemented for sequential observations.")

    activation_class = get_activation_class(config["actor"]["activation_function"])

    input_dim = (env.observation_spec[("agents", "observation")].shape[-1] 
                 if not config["flex_encoder"]["enabled"] 
                 else config["flex_encoder"]["output_dim"])

    policy_net = MultiAgentMLP(
        n_agent_inputs=input_dim,
        n_agent_outputs=env.full_action_spec[("agents", "action")].shape[-1],
        n_agents=config["environment"]["num_drones"],
        centralised=config["actor"]["centralized"],
        share_params=config["actor"]["share_parameters"],
        device=device,
        depth=config["actor"]["network_depth"],
        num_cells=config["actor"]["network_width"],
        activation_class=activation_class
    )

    # Uses encoded observations as input if flex encoder is enabled
    if config["flex_encoder"]["enabled"]:
        flex_encoder = create_flex_encoder(env, config, device, out_key=("agents", "encoded_obs"))
        policy_module = TensorDictModule(
            policy_net,
            in_keys=[("agents", "encoded_obs")],
            out_keys=[("agents", "param")],
        )
        return TensorDictSequential(flex_encoder, policy_module)
    # Else, uses raw observations as input
    else:
        policy_module = TensorDictModule(
            policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "param")],
        )
        return policy_module

def create_actor(
    env: EnvBase,
    device: torch.device,
    config: Dict[str, Any],
) -> ProbabilisticActor:
    """Create the deterministic (Tanh-squashed delta) multi-agent actor.

    Args:
        env: TorchRL environment providing observation and action specs.
        device: Target device for modules.
        config: Hierarchical configuration dictionary.

    Returns:
        ProbabilisticActor: Actor producing actions under a TanhDelta distribution.
    """


    policy_module = create_policy_module(env, config, device)

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
    actor: ProbabilisticActor,
    device: torch.device,
    config: Dict[str, Any],
) -> tuple[TensorDictModule, AdditiveGaussianModule]:
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
    config: Dict[str, Any],
) -> ProbabilisticActor:
    """Create a multi-agent PPO actor with a TanhNormal squashed Gaussian policy.

    Returns a ProbabilisticActor that emits actions within the environment bounds and stores
    the log probability under the key ("agents", "sample_log_prob") for PPO updates.
    """
    activation_class = get_activation_class(config["actor"]["activation_function"])
    action_dim = env.full_action_spec[("agents", "action")].shape[-1]

    # Policy backbone produces concatenated normal params (mean, std) per action dim
    policy_backbone = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec[("agents", "observation")].shape[-1],
            n_agent_outputs=action_dim * 2,
            n_agents=config["environment"]["num_drones"],
            centralised=config["actor"]["centralized"],
            share_params=config["actor"]["share_parameters"],
            device=device,
            depth=config["actor"]["network_depth"],
            num_cells=config["actor"]["network_width"],
            activation_class=activation_class,
        ),
        NormalParamExtractor()
    )

    policy_module = TensorDictModule(
        module=policy_backbone,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    # Squashed Gaussian policy with log-prob output
    policy = ProbabilisticActor(
        module=policy_module,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        spec=env.full_action_spec[("agents", "action")],
        distribution_class=TanhNormal,
        # TanhNormal will be created from loc/scale; env spec handles bounds
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    return policy