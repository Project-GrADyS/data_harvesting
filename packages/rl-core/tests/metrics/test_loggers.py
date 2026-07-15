from __future__ import annotations

from io import StringIO
import sys

import mlflow
import pytest

import rl_core
import rl_core.metrics as metrics
from rl_core.metrics import ConsoleMetricLogger, MetricLogger, MLflowMetricLogger


def test_logger_public_api_is_exported_from_package_root() -> None:
    assert rl_core.ConsoleMetricLogger is ConsoleMetricLogger
    assert rl_core.MLflowMetricLogger is MLflowMetricLogger
    assert rl_core.MetricLogger is MetricLogger
    assert metrics.ConsoleMetricLogger is ConsoleMetricLogger
    assert metrics.MLflowMetricLogger is MLflowMetricLogger
    assert metrics.MetricLogger is MetricLogger


def test_console_logger_writes_sorted_single_line() -> None:
    stream = StringIO()
    logger = ConsoleMetricLogger(prefix="train", stream=stream)

    logger({"reward": 3.5, "loss": 0.25}, step=12)

    assert stream.getvalue() == "train step=12: loss=0.25, reward=3.5\n"


def test_console_logger_can_omit_prefix(capsys: pytest.CaptureFixture[str]) -> None:
    ConsoleMetricLogger(prefix=None)({"reward": 1.0}, step=2)

    assert capsys.readouterr().out == "step=2: reward=1.0\n"


def test_console_logger_formats_empty_snapshot() -> None:
    stream = StringIO()

    ConsoleMetricLogger(stream=stream)({}, step=0)

    assert stream.getvalue() == "metrics step=0: \n"


def test_console_logger_uses_current_stdout_at_call_time(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = ConsoleMetricLogger()

    logger({"reward": 1.0}, step=2)

    assert capsys.readouterr().out == "metrics step=2: reward=1.0\n"


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


def test_mlflow_logger_does_not_modify_the_callers_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, float]] = []
    monkeypatch.setattr(
        mlflow,
        "log_metrics",
        lambda metrics, *, step: payloads.append(metrics),
    )
    source = {"reward": 4.0}

    MLflowMetricLogger(prefix="evaluation")(source, step=8)

    assert source == {"reward": 4.0}
    assert payloads == [{"evaluation/reward": 4.0}]


def test_mlflow_logger_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)

    with pytest.raises(ModuleNotFoundError, match="'mlflow' extra"):
        MLflowMetricLogger()({"reward": 4.0}, step=8)


@pytest.mark.parametrize("logger_type", [ConsoleMetricLogger, MLflowMetricLogger])
def test_loggers_reject_empty_prefix(logger_type: type[object]) -> None:
    with pytest.raises(ValueError, match="prefix"):
        logger_type(prefix="")


@pytest.mark.parametrize("logger_type", [ConsoleMetricLogger, MLflowMetricLogger])
@pytest.mark.parametrize("prefix", [False, 1, object()])
def test_loggers_reject_non_string_prefix(logger_type: type[object], prefix: object) -> None:
    with pytest.raises(TypeError, match="prefix"):
        logger_type(prefix=prefix)
