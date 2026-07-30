from __future__ import annotations

import torch

from kaz_training.environment import make_env
from kaz_training.evaluation import evaluate
from kaz_training.models import create_actor


def test_seeded_evaluation_is_repeatable(kaz_config) -> None:
    env = make_env(kaz_config)
    try:
        actor = create_actor(env, kaz_config, torch.device("cpu"))
    finally:
        env.close()

    first = evaluate(actor, kaz_config, num_episodes=1, seed=123)
    second = evaluate(actor, kaz_config, num_episodes=1, seed=123)

    assert first == second
