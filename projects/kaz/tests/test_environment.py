from __future__ import annotations

import torch
from torchrl.envs import check_env_specs

from kaz_training.environment import make_env, max_entities


def test_kaz_sequence_observations_are_padded_and_masked(kaz_config) -> None:
    env = make_env(kaz_config)
    try:
        check_env_specs(env)
        reset = env.reset()
        entities = reset["agents", "observation", "entities"]
        entity_mask = reset["agents", "observation", "entity_mask"]
        role = reset["agents", "observation", "role"]

        assert entities.shape == torch.Size(
            [2, max_entities(kaz_config["environment"]), 11]
        )
        assert entity_mask.shape == entities.shape[:-1]
        assert entity_mask.dtype is torch.bool
        assert torch.all(entities[~entity_mask] == 0)
        assert torch.equal(role[0], torch.tensor([1.0, 0.0]))
        assert torch.equal(role[1], torch.tensor([0.0, 1.0]))
        assert reset["agents", "mask"].all()
    finally:
        env.close()


def test_kaz_rollout_exposes_team_episode_metrics(kaz_config) -> None:
    env = make_env(kaz_config)
    try:
        rollout = env.rollout(max_steps=kaz_config["environment"]["max_cycles"])
        assert ("next", "team_reward") in rollout.keys(include_nested=True)
        assert ("next", "episode_team_reward") in rollout.keys(include_nested=True)
        assert ("next", "step_count") in rollout.keys(include_nested=True)
        assert rollout["next", "team_reward"].shape[-1] == 1
    finally:
        env.close()
