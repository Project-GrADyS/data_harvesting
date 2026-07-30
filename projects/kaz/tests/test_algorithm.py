from __future__ import annotations

import torch

from kaz_training.algorithm import MADDPG
from kaz_training.environment import make_env


def _parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before, module) -> bool:
    return any(
        not torch.allclose(old, new)
        for old, new in zip(before, module.parameters(), strict=True)
    )


def _tensordict_parameters(parameters):
    return [
        value.detach().clone()
        for value in parameters.flatten_keys().values()
    ]


def test_maddpg_warmup_and_learning_update_models(kaz_config) -> None:
    env = make_env(kaz_config)
    try:
        algorithm = MADDPG(env, kaz_config, torch.device("cpu"))
        batch = env.rollout(
            max_steps=8,
            policy=algorithm.exploratory_policy,
        ).reshape(-1)
        actor_before = _parameters(algorithm.policy)
        critic_before = _parameters(algorithm.critic)
        target_actor_before = _tensordict_parameters(
            algorithm.loss_module.target_actor_network_params
        )
        target_critic_before = _tensordict_parameters(
            algorithm.loss_module.target_value_network_params
        )
        epsilon_before = algorithm.epsilon

        losses = algorithm.learn(batch)

        assert set(losses) == {"loss_actor", "loss_value"}
        assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
        assert _changed(actor_before, algorithm.policy)
        assert _changed(critic_before, algorithm.critic)
        assert any(
            not torch.allclose(old, new)
            for old, new in zip(
                target_actor_before,
                algorithm.loss_module.target_actor_network_params.flatten_keys().values(),
                strict=True,
            )
        )
        assert any(
            not torch.allclose(old, new)
            for old, new in zip(
                target_critic_before,
                algorithm.loss_module.target_value_network_params.flatten_keys().values(),
                strict=True,
            )
        )
        assert algorithm.epsilon < epsilon_before
    finally:
        env.close()


def test_maddpg_does_not_sample_before_warmup(kaz_config) -> None:
    kaz_config["training"]["warmup_steps"] = 32
    env = make_env(kaz_config)
    try:
        algorithm = MADDPG(env, kaz_config, torch.device("cpu"))
        batch = env.rollout(
            max_steps=4,
            policy=algorithm.exploratory_policy,
        ).reshape(-1)
        assert algorithm.learn(batch) == {}
        assert len(algorithm.replay_buffer) == batch.numel()
    finally:
        env.close()
