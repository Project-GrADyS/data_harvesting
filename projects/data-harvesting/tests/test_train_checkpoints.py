from types import SimpleNamespace

import torch
from torch import nn

from data_harvesting.train import _configure_scheduler, _should_save_final_model, log_model


class _TrackingPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.to_calls: list[str] = []

    def to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if target is not None:
            self.to_calls.append(str(target))
        return super().to(*args, **kwargs)


def _config(*, checkpoint_enabled: bool = True) -> dict:
    return {
        "metrics": {"log_every_n_steps": 50},
        "checkpoint": {
            "enabled": checkpoint_enabled,
            "checkpoint_every_n_steps": 100,
            "save_final_model": True,
        },
        "evaluation": {
            "enabled": True,
            "eval_every_n_steps": 200,
        },
    }


def test_log_model_logs_cpu_copy_without_mutating_live_policy(monkeypatch) -> None:
    logged: list[tuple[nn.Module, str]] = []
    policy = _TrackingPolicy()
    monkeypatch.setattr(
        "data_harvesting.train.mlflow_pytorch.log_model",
        lambda model, name: logged.append((model, name)),
    )

    log_model(SimpleNamespace(policy=policy), name="policy_checkpoint_step_100")

    logged_model, logged_name = logged[0]
    assert logged_name == "policy_checkpoint_step_100"
    assert logged_model is not policy
    assert logged_model.to_calls == ["cpu"]
    assert policy.to_calls == []


def test_training_scheduler_dispatches_project_callbacks_at_frame_boundaries() -> None:
    calls: list[tuple[str, int]] = []
    scheduler = _configure_scheduler(
        _config(),
        metrics_callback=lambda step: calls.append(("metrics", step)),
        checkpoint_callback=lambda step: calls.append(("checkpoint", step)),
        evaluation_callback=lambda step: calls.append(("evaluation", step)),
    )

    scheduler.step(60)
    scheduler.step(60)
    scheduler.step(100)

    assert calls == [
        ("metrics", 60),
        ("metrics", 120),
        ("checkpoint", 120),
        ("metrics", 220),
        ("checkpoint", 220),
        ("evaluation", 220),
    ]


def test_training_scheduler_omits_disabled_checkpoint_callback() -> None:
    calls: list[int] = []
    scheduler = _configure_scheduler(
        _config(checkpoint_enabled=False),
        metrics_callback=lambda step: None,
        checkpoint_callback=calls.append,
        evaluation_callback=lambda step: None,
    )

    scheduler.step(500)

    assert calls == []


def test_should_save_final_model_uses_checkpoint_config() -> None:
    config = _config()
    assert _should_save_final_model(config)
    config["checkpoint"]["save_final_model"] = False
    assert not _should_save_final_model(config)
