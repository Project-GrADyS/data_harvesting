from collections.abc import Mapping
import sys
from typing import Protocol, TextIO


class MetricLogger(Protocol):
    """A destination for one snapshot of aggregated metrics."""

    def __call__(self, metrics: Mapping[str, float], *, step: int) -> None: ...


class ConsoleMetricLogger:
    """Write each metric snapshot as a deterministic single line."""

    def __init__(self, *, prefix: str | None = "metrics", stream: TextIO | None = None) -> None:
        if prefix is not None and (not isinstance(prefix, str) or not prefix):
            raise ValueError("prefix must be a non-empty string or None.")
        self._prefix = prefix
        self._stream = stream

    def __call__(self, metrics: Mapping[str, float], *, step: int) -> None:
        fields = ", ".join(f"{key}={metrics[key]}" for key in sorted(metrics))
        header = f"{self._prefix} " if self._prefix is not None else ""
        print(f"{header}step={step}: {fields}", file=self._stream or sys.stdout)


class MLflowMetricLogger:
    """Log metric snapshots to the currently active MLflow run."""

    def __init__(self, *, prefix: str | None = None) -> None:
        if prefix is not None and (not isinstance(prefix, str) or not prefix):
            raise ValueError("prefix must be a non-empty string or None.")
        self._prefix = prefix

    def __call__(self, metrics: Mapping[str, float], *, step: int) -> None:
        try:
            from mlflow import log_metrics
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "MLflowMetricLogger requires the optional MLflow dependency. "
                "Install rl-core with the 'mlflow' extra."
            ) from error

        if self._prefix is None:
            payload = dict(metrics)
        else:
            payload = {f"{self._prefix}/{key}": value for key, value in metrics.items()}
        log_metrics(payload, step=step)
