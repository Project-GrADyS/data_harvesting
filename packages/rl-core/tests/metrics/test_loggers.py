from __future__ import annotations

from io import StringIO

import mlflow
import pytest

import rl_core
from rl_core.metrics import ConsoleMetricLogger, MLflowMetricLogger


def test_logger_public_api_is_exported_from_package_root() -> None:
    assert rl_core.ConsoleMetricLogger is ConsoleMetricLogger
    assert rl_core.MLflowMetricLogger is MLflowMetricLogger


def test_console_logger_writes_sorted_single_line() -> None:
    stream = StringIO()
    logger = ConsoleMetricLogger(prefix="train", stream=stream)

    logger({"reward": 3.5, "loss": 0.25}, step=12)

    assert stream.getvalue() == "train step=12: loss=0.25, reward=3.5\n"


def test_console_logger_can_omit_prefix(capsys: pytest.CaptureFixture[str]) -> None:
    ConsoleMetricLogger(prefix=None)({"reward": 1.0}, step=2)

    assert capsys.readouterr().out == "step=2: reward=1.0\n"


def test_mlflow_logger_forwards_metrics_and_step(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[dict[str, float], int]] = []
    monkeypatch.setattr(mlflow, "log_metrics", lambda metrics, *, step: calls.append((metrics, step)))

    MLflowMetricLogger()({"reward": 4.0}, step=8)

    assert calls == [({"reward": 4.0}, 8)]


def test_mlflow_logger_applies_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[dict[str, float], int]] = []
    monkeypatch.setattr(mlflow, "log_metrics", lambda metrics, *, step: calls.append((metrics, step)))

    MLflowMetricLogger(prefix="evaluation")({"reward": 4.0, "length": 10.0}, step=8)

    assert calls == [({"evaluation/reward": 4.0, "evaluation/length": 10.0}, 8)]


@pytest.mark.parametrize("logger_type", [ConsoleMetricLogger, MLflowMetricLogger])
def test_loggers_validate_prefix(logger_type) -> None:
    with pytest.raises(ValueError, match="prefix"):
        logger_type(prefix="")
