from types import SimpleNamespace

import torch
from rl_core import ScalarMetricSpec, ScalarReducer
from torch import nn

from data_harvesting.train import _SeededRolloutEnvironment, _run_periodic_evaluation


class _FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)


class _FakeEnv:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.closed = False

    def set_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def rollout(self, *args, **kwargs):
        return SimpleNamespace()

    def close(self) -> None:
        self.closed = True


def test_seeded_rollout_environment_owns_deterministic_seed_progression() -> None:
    environment = _FakeEnv()
    seeded = _SeededRolloutEnvironment(environment, 7)

    seeded.rollout()
    seeded.rollout()
    seeded.rollout()
    seeded.close()

    assert environment.seeds == [7, 8, 9]
    assert environment.closed


def test_periodic_evaluation_builds_rl_core_evaluator_with_cpu_policy(monkeypatch) -> None:
    policy = _FakePolicy()
    algorithm = SimpleNamespace(policy=policy)
    captured = {}

    class _FakeEvaluator:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def run(self, step: int):
            captured["step"] = step
            return {"avg_reward": 2.0}

    monkeypatch.setattr("data_harvesting.train.Evaluator", _FakeEvaluator)

    result = _run_periodic_evaluation(
        algorithm,
        {"environment": {"max_episode_length": 3}},
        experience_steps=120,
        metrics_spec=(
            ScalarMetricSpec(key="avg_reward", reducer=ScalarReducer.MEAN),
        ),
        num_runs=4,
        seed=11,
    )

    assert result == {"avg_reward": 2.0}
    assert captured["step"] == 120
    assert captured["config"].num_episodes == 4
    assert captured["config"].max_steps == 3
    assert captured["policy"] is not policy
    assert next(captured["policy"].parameters()).device.type == "cpu"
