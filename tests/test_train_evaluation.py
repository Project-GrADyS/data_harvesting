from types import SimpleNamespace

import pytest
import torch
from torch import nn

from data_harvesting.train import _maybe_run_periodic_evaluation, _run_periodic_evaluation


class _FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.calls = 0

    def forward(self, tensordict):
        self.calls += 1
        return tensordict


class _ExploratoryPolicy(nn.Module):
    def forward(self, tensordict):
        raise AssertionError("periodic evaluation must not use exploratory_policy")


class _FakeAlgorithm:
    def __init__(self) -> None:
        self.policy = _FakePolicy()
        self.exploratory_policy = _ExploratoryPolicy()


class _FakeRollout:
    def __init__(self, marker: int) -> None:
        self.marker = marker

    def reshape(self, *shape):
        return self


class _FakeEnv:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.policies: list[nn.Module] = []
        self.closed = False

    def to(self, device: torch.device):
        return self

    def set_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def rollout(self, *, max_steps: int, policy: nn.Module):
        assert max_steps == 3
        assert policy.training is False
        self.policies.append(policy)
        policy(SimpleNamespace())
        return _FakeRollout(len(self.policies))

    def close(self) -> None:
        self.closed = True


class _FakeMetricsCollector:
    instances: list["_FakeMetricsCollector"] = []

    def __init__(self, device: torch.device, metrics_spec) -> None:
        self.reported: list[_FakeRollout] = []
        self.__class__.instances.append(self)

    def report_metrics(self, batch: _FakeRollout) -> None:
        self.reported.append(batch)

    def _build_log_metrics(self) -> dict[str, float]:
        return {
            "avg_reward": float(len(self.reported)),
            "end_cause_STALLED": 2.0,
        }


def _evaluation_config() -> dict:
    return {
        "environment": {
            "max_episode_length": 3,
        },
        "evaluation": {
            "enabled": True,
            "eval_every_n_steps": 100,
            "num_runs": 4,
            "seed": 7,
        },
    }


def test_run_periodic_evaluation_uses_eval_mode_policy_and_prefixed_metrics(monkeypatch) -> None:
    algorithm = _FakeAlgorithm()
    algorithm.policy.train(True)
    env = _FakeEnv()
    logged: list[tuple[dict[str, float], int]] = []
    _FakeMetricsCollector.instances = []

    monkeypatch.setattr("data_harvesting.train.make_env", lambda config: env)
    monkeypatch.setattr("data_harvesting.train.EnvironmentMetricsCollector", _FakeMetricsCollector)
    monkeypatch.setattr("data_harvesting.train.mlflow.log_metrics", lambda metrics, step: logged.append((metrics, step)))

    _run_periodic_evaluation(
        algorithm,
        _evaluation_config(),
        experience_steps=120,
        device=torch.device("cpu"),
        metrics_spec=SimpleNamespace(),
        num_runs=4,
        seed=7,
    )

    assert algorithm.policy.training is True
    assert algorithm.policy.calls == 4
    assert env.policies == [algorithm.policy] * 4
    assert env.seeds == [7, 8, 9, 10]
    assert env.closed
    assert [rollout.marker for rollout in _FakeMetricsCollector.instances[0].reported] == [1, 2, 3, 4]
    assert logged == [
        (
            {
                "eval/avg_reward": 4.0,
                "eval/end_cause_STALLED": 2.0,
            },
            120,
        )
    ]
    assert all(key.startswith("eval/") for key in logged[0][0])


def test_run_periodic_evaluation_restores_eval_state(monkeypatch) -> None:
    algorithm = _FakeAlgorithm()
    algorithm.policy.eval()
    _FakeMetricsCollector.instances = []

    monkeypatch.setattr("data_harvesting.train.make_env", lambda config: _FakeEnv())
    monkeypatch.setattr("data_harvesting.train.EnvironmentMetricsCollector", _FakeMetricsCollector)
    monkeypatch.setattr("data_harvesting.train.mlflow.log_metrics", lambda metrics, step: None)

    _run_periodic_evaluation(
        algorithm,
        _evaluation_config(),
        experience_steps=120,
        device=torch.device("cpu"),
        metrics_spec=SimpleNamespace(),
        num_runs=1,
        seed=None,
    )

    assert algorithm.policy.training is False


def test_maybe_run_periodic_evaluation_respects_interval(monkeypatch) -> None:
    algorithm = _FakeAlgorithm()
    calls: list[tuple[int, int, int | None]] = []

    def _fake_run_periodic_evaluation(algorithm, config, *, experience_steps, device, metrics_spec, num_runs, seed):
        calls.append((experience_steps, num_runs, seed))

    monkeypatch.setattr("data_harvesting.train._run_periodic_evaluation", _fake_run_periodic_evaluation)

    last_eval_step = _maybe_run_periodic_evaluation(
        algorithm,
        _evaluation_config(),
        experience_steps=50,
        last_eval_step=0,
        device=torch.device("cpu"),
        metrics_spec=SimpleNamespace(),
    )
    assert last_eval_step == 0
    assert calls == []

    last_eval_step = _maybe_run_periodic_evaluation(
        algorithm,
        _evaluation_config(),
        experience_steps=100,
        last_eval_step=0,
        device=torch.device("cpu"),
        metrics_spec=SimpleNamespace(),
    )
    assert last_eval_step == 100
    assert calls == [(100, 4, 7)]


@pytest.mark.parametrize(
    "evaluation_config",
    [
        {"enabled": False, "eval_every_n_steps": 100, "num_runs": 4},
        {"enabled": True, "eval_every_n_steps": 0, "num_runs": 4},
    ],
)
def test_maybe_run_periodic_evaluation_can_be_disabled(monkeypatch, evaluation_config) -> None:
    calls: list[int] = []
    config = _evaluation_config()
    config["evaluation"] = evaluation_config

    monkeypatch.setattr(
        "data_harvesting.train._run_periodic_evaluation",
        lambda *args, **kwargs: calls.append(kwargs["experience_steps"]),
    )

    last_eval_step = _maybe_run_periodic_evaluation(
        _FakeAlgorithm(),
        config,
        experience_steps=100,
        last_eval_step=0,
        device=torch.device("cpu"),
        metrics_spec=SimpleNamespace(),
    )

    assert last_eval_step == 0
    assert calls == []
