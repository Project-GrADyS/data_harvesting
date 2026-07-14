from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from numbers import Real
from typing import TypeAlias

import torch

from .loggers import MetricLogger
from .specs import CategoricalMetricSpec, MetricSpec, ScalarMetricSpec, ScalarReducer

from tensordict import TensorDictBase


MetricValue: TypeAlias = torch.Tensor | Real | Sequence[Real]
MetricValues: TypeAlias = Mapping[str, MetricValue] | TensorDictBase


class MetricsCollector:
    """Accumulate configured metrics and publish snapshots to caller-provided loggers."""

    def __init__(
        self,
        *,
        specs: Iterable[MetricSpec],
        loggers: Iterable[MetricLogger] = (),
        device: torch.device | str | None = None,
    ) -> None:
        self._specs = tuple(specs)
        if not self._specs:
            raise ValueError("specs must contain at least one metric specification.")
        if not all(isinstance(spec, (ScalarMetricSpec, CategoricalMetricSpec)) for spec in self._specs):
            raise TypeError("Every metric specification must be a ScalarMetricSpec or CategoricalMetricSpec.")

        keys = [spec.key for spec in self._specs]
        if len(keys) != len(set(keys)):
            raise ValueError("Metric specification keys must be unique.")
        self._specs_by_key = {spec.key: spec for spec in self._specs}
        self._validate_output_names()

        self._loggers = tuple(loggers)
        if not all(callable(logger) for logger in self._loggers):
            raise TypeError("Every logger must be callable.")
        self._device = torch.device(device) if device is not None else torch.device("cpu")
        self.reset()

    def push(self, values: MetricValues) -> None:
        """Accumulate one or more configured metrics.

        Missing configured metrics are allowed, enabling metrics with different sample counts.
        """

        items = tuple(self._iter_items(values))

        for key, value in items:
            if key not in self._specs_by_key:
                continue # Ignore unknown keys
            spec = self._specs_by_key[key]
            tensor = self._as_tensor(key, value)
            if tensor.numel() == 0:
                continue
            if isinstance(spec, ScalarMetricSpec):
                self._push_scalar(spec, tensor)
            else:
                self._push_categorical(spec, tensor)

    def peek(self) -> dict[str, float]:
        """Return current aggregates without modifying collector state."""

        result: dict[str, float] = {}
        for spec in self._specs:
            if isinstance(spec, ScalarMetricSpec):
                count = self._scalar_counts[spec.key]
                if count == 0:
                    continue
                total = self._scalar_totals[spec.key]
                value = total / count if spec.reducer is ScalarReducer.MEAN else total
                result[spec.resolved_output_name] = float(value.item())
                continue

            if not self._categorical_seen[spec.key]:
                continue
            prefix = spec.resolved_output_prefix
            for label, count in self._categorical_counts[spec.key].items():
                result[f"{prefix}_{label}"] = float(count.item())
        return result

    def flush(self, *, step: int) -> dict[str, float]:
        """Publish a snapshot and reset after every logger succeeds.

        Logger exceptions propagate and leave all accumulated state intact.
        """

        metrics = self.peek()
        if not metrics:
            return metrics
        for logger in self._loggers:
            logger(dict(metrics), step=step)
        self.reset()
        return metrics

    def reset(self) -> None:
        """Discard every accumulated metric value."""

        self._scalar_totals = {
            spec.key: torch.zeros((), dtype=torch.float64, device=self._device)
            for spec in self._specs
            if isinstance(spec, ScalarMetricSpec)
        }
        self._scalar_counts = {
            spec.key: 0 for spec in self._specs if isinstance(spec, ScalarMetricSpec)
        }
        self._categorical_counts = {
            spec.key: {
                label: torch.zeros((), dtype=torch.int64, device=self._device)
                for label in spec.value_labels.values()
            }
            for spec in self._specs
            if isinstance(spec, CategoricalMetricSpec)
        }
        self._categorical_seen = {
            spec.key: False for spec in self._specs if isinstance(spec, CategoricalMetricSpec)
        }

    def _validate_output_names(self) -> None:
        output_names: list[str] = []
        for spec in self._specs:
            if isinstance(spec, ScalarMetricSpec):
                output_names.append(spec.resolved_output_name)
            else:
                output_names.extend(
                    f"{spec.resolved_output_prefix}_{label}" for label in spec.value_labels.values()
                )
        if len(output_names) != len(set(output_names)):
            raise ValueError("Metric specifications produce duplicate output names.")

    def _iter_items(self, values: MetricValues) -> Iterator[tuple[str, MetricValue]]:
        if isinstance(values, Mapping):
            yield from values.items()
            return

        try:
            from tensordict import TensorDictBase
        except ModuleNotFoundError:
            TensorDictBase = None  # type: ignore[assignment,misc]

        if TensorDictBase is not None and isinstance(values, TensorDictBase):
            keys = values.keys(include_nested=False, leaves_only=True)
            for key in keys:
                if not isinstance(key, str):
                    raise TypeError("TensorDict metric keys must be top-level strings.")
                yield key, values.get(key)
            return

        raise TypeError("values must be a mapping or a TensorDictBase from the optional tensordict dependency.")

    def _as_tensor(self, key: str, value: MetricValue) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device=self._device)
        elif isinstance(value, Real) and not isinstance(value, bool):
            tensor = torch.as_tensor(value, device=self._device)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            tensor = torch.as_tensor(value, device=self._device)
        else:
            raise TypeError(f"Metric {key!r} must be a tensor or numeric value, got {type(value)}.")
        if tensor.dtype is torch.bool or tensor.is_complex():
            raise TypeError(f"Metric {key!r} must contain real numeric values, got dtype {tensor.dtype}.")
        return tensor.reshape(-1)

    def _push_scalar(self, spec: ScalarMetricSpec, tensor: torch.Tensor) -> None:
        if not tensor.is_floating_point() and tensor.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError(f"Scalar metric {spec.key!r} must contain numeric values, got {tensor.dtype}.")
        self._scalar_totals[spec.key] += tensor.to(torch.float64).sum()
        self._scalar_counts[spec.key] += tensor.numel()

    def _push_categorical(self, spec: CategoricalMetricSpec, tensor: torch.Tensor) -> None:
        if tensor.is_floating_point():
            if not bool(torch.all(tensor == tensor.round())):
                raise ValueError(f"Categorical metric {spec.key!r} must contain integer-valued samples.")
            tensor = tensor.to(torch.int64)
        elif tensor.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            tensor = tensor.to(torch.int64)
        else:
            raise TypeError(f"Categorical metric {spec.key!r} must contain integer values, got {tensor.dtype}.")

        allowed = torch.tensor(tuple(spec.value_labels), dtype=torch.int64, device=self._device)
        unknown = tensor[~torch.isin(tensor, allowed)]
        if unknown.numel():
            unknown_values = sorted(set(unknown.cpu().tolist()))
            raise ValueError(f"Categorical metric {spec.key!r} contains unknown values: {unknown_values}")

        for raw_value, label in spec.value_labels.items():
            self._categorical_counts[spec.key][label] += (tensor == raw_value).sum()
        self._categorical_seen[spec.key] = True
