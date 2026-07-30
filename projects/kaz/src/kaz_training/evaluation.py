from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from kaz_training.environment import make_env


def evaluate(
    policy: torch.nn.Module,
    config: Mapping[str, Any],
    *,
    num_episodes: int,
    seed: int | None,
) -> dict[str, float]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    returns: list[float] = []
    lengths: list[float] = []
    environment = make_env(config)
    was_training = policy.training
    try:
        policy.eval()
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for episode in range(num_episodes):
                if seed is not None:
                    environment.set_seed(seed + episode)
                rollout = environment.rollout(
                    max_steps=int(config["environment"]["max_cycles"]),
                    policy=policy,
                    break_when_any_done=True,
                )
                done = rollout.get(("next", "done")).squeeze(-1).to(torch.bool)
                if not bool(done.any()):
                    raise RuntimeError("Evaluation reached max_cycles without a terminal step.")
                terminal = rollout[done][-1]
                returns.append(float(terminal.get(("next", "episode_team_reward")).item()))
                lengths.append(float(terminal.get(("next", "step_count")).item()))
    finally:
        policy.train(was_training)
        environment.close()

    return {
        "team_kills_mean": float(np.mean(returns)),
        "team_kills_std": float(np.std(returns)),
        "episode_length_mean": float(np.mean(lengths)),
    }
