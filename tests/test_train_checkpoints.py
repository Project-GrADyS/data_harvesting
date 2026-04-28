from types import SimpleNamespace

import torch
from torch import nn

from data_harvesting.train import _maybe_log_checkpoint, _should_save_final_model, log_model, train


class _TrackingPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.to_calls: list[str] = []

    def to(self, *args, **kwargs):
        if args:
            self.to_calls.append(str(args[0]))
        elif "device" in kwargs:
            self.to_calls.append(str(kwargs["device"]))
        return super().to(*args, **kwargs)


def _checkpoint_config(*, enabled: bool, checkpoint_every_n_steps: int, save_final_model: bool = True) -> dict:
    return {
        "checkpoint": {
            "enabled": enabled,
            "checkpoint_every_n_steps": checkpoint_every_n_steps,
            "save_final_model": save_final_model,
        }
    }


def test_maybe_log_checkpoint_respects_disabled_checkpointing(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("data_harvesting.train.log_model", lambda algorithm, name="policy_model": calls.append(name))

    last_checkpoint_step = _maybe_log_checkpoint(
        SimpleNamespace(policy=_TrackingPolicy()),
        _checkpoint_config(enabled=False, checkpoint_every_n_steps=100),
        experience_steps=500,
        last_checkpoint_step=0,
    )

    assert last_checkpoint_step == 0
    assert calls == []


def test_maybe_log_checkpoint_uses_experience_step_interval(monkeypatch) -> None:
    calls: list[str] = []
    algorithm = SimpleNamespace(policy=_TrackingPolicy())
    config = _checkpoint_config(enabled=True, checkpoint_every_n_steps=100)

    monkeypatch.setattr("data_harvesting.train.log_model", lambda algorithm, name="policy_model": calls.append(name))

    last_checkpoint_step = _maybe_log_checkpoint(
        algorithm,
        config,
        experience_steps=60,
        last_checkpoint_step=0,
    )
    assert last_checkpoint_step == 0
    assert calls == []

    last_checkpoint_step = _maybe_log_checkpoint(
        algorithm,
        config,
        experience_steps=120,
        last_checkpoint_step=last_checkpoint_step,
    )
    assert last_checkpoint_step == 120
    assert calls == ["policy_checkpoint_step_120"]

    last_checkpoint_step = _maybe_log_checkpoint(
        algorithm,
        config,
        experience_steps=180,
        last_checkpoint_step=last_checkpoint_step,
    )
    assert last_checkpoint_step == 120
    assert calls == ["policy_checkpoint_step_120"]


def test_log_model_logs_cpu_copy_without_mutating_live_policy(monkeypatch) -> None:
    logged: list[tuple[nn.Module, str]] = []
    policy = _TrackingPolicy()
    algorithm = SimpleNamespace(policy=policy)

    monkeypatch.setattr("data_harvesting.train.mlflow_pytorch.log_model", lambda model, name: logged.append((model, name)))

    log_model(algorithm, name="policy_checkpoint_step_100")

    assert len(logged) == 1
    logged_model, logged_name = logged[0]
    assert logged_name == "policy_checkpoint_step_100"
    assert logged_model is not policy
    assert isinstance(logged_model, _TrackingPolicy)
    assert logged_model.to_calls == ["cpu"]
    assert policy.to_calls == []


def test_should_save_final_model_uses_checkpoint_config() -> None:
    assert _should_save_final_model(_checkpoint_config(enabled=False, checkpoint_every_n_steps=0, save_final_model=True))
    assert not _should_save_final_model(
        _checkpoint_config(enabled=False, checkpoint_every_n_steps=0, save_final_model=False)
    )


def test_train_logs_periodic_and_final_models_from_loop(monkeypatch) -> None:
    logged_models: list[str] = []

    class _FakeBatch:
        def __init__(self, frames: int) -> None:
            self._frames = frames

        def numel(self) -> int:
            return self._frames

        def reshape(self, *shape):
            return self

    class _FakeCollector:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter([_FakeBatch(60), _FakeBatch(60)])

        def update_policy_weights_(self) -> None:
            return None

    class _FakeMetricLogger:
        def __init__(self, *args, **kwargs) -> None:
            self.scalar_totals = {"avg_reward": torch.tensor(1.0)}

        def report_metrics(self, batch) -> None:
            return None

        def log_metrics(self, step: int) -> None:
            return None

        def metric_value(self, key: str) -> float:
            return 1.0

    class _FakeLearningLogger:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def report_loss(self, loss_name: str, loss_value: torch.Tensor) -> None:
            return None

        def log_metrics(self, step: int) -> None:
            return None

    class _FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeAlgorithm:
        def __init__(self, env, device, config) -> None:
            self.policy = _TrackingPolicy()
            self.exploratory_policy = self.policy

        def learn(self, batch) -> dict[str, torch.Tensor]:
            return {"actor": torch.tensor(1.0)}

    config = {
        "training": {
            "algorithm": "maddpg",
            "total_timesteps": 120,
        },
        "metrics": {
            "log_every_n_steps": 1000,
        },
        "collector": {
            "device": "cpu",
        },
        "checkpoint": {
            "enabled": True,
            "checkpoint_every_n_steps": 100,
            "save_final_model": True,
        },
    }

    monkeypatch.setattr("data_harvesting.train.make_metrics_spec", lambda: SimpleNamespace())
    monkeypatch.setattr(
        "data_harvesting.train.make_env",
        lambda config: SimpleNamespace(reward_keys=[], group_map={}),
    )
    monkeypatch.setattr("data_harvesting.train.TransformedEnv", lambda base_env, transform: base_env)
    monkeypatch.setattr("data_harvesting.train.RewardSum", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr("data_harvesting.train.check_env_specs", lambda env: None)
    monkeypatch.setattr("data_harvesting.train.MADDPGAlgorithm", _FakeAlgorithm)
    monkeypatch.setattr("data_harvesting.train.create_collector", lambda *args, **kwargs: _FakeCollector())
    monkeypatch.setattr("data_harvesting.train.EnvironmentMetricsCollector", _FakeMetricLogger)
    monkeypatch.setattr("data_harvesting.train.LearningMetricsCollector", _FakeLearningLogger)
    monkeypatch.setattr("data_harvesting.train.mlflow.start_run", lambda run_name=None: _FakeRun())
    monkeypatch.setattr("data_harvesting.train.mlflow.log_params", lambda config: None)
    monkeypatch.setattr("data_harvesting.train.log_model", lambda algorithm, name="policy_model": logged_models.append(name))
    monkeypatch.setattr("data_harvesting.train.tqdm", lambda total: SimpleNamespace(update=lambda frames: None))

    result = train(config)

    assert result == 1.0
    assert logged_models == ["policy_checkpoint_step_120", "policy_model"]
