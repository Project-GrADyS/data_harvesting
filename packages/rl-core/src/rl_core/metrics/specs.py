from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from validation_core import validate_mapping, validate_non_empty_string


class ScalarReducer(StrEnum):
    """Supported reductions for scalar metric samples."""

    MEAN = "mean"
    SUM = "sum"


@dataclass(frozen=True, slots=True, kw_only=True)
class ScalarMetricSpec:
    """Declare a numeric metric reduced across every pushed sample."""

    key: str
    reducer: ScalarReducer
    output_name: str | None = None

    @property
    def resolved_output_name(self) -> str:
        return self.output_name or self.key


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoricalMetricSpec:
    """Declare a finite categorical metric expanded into one count per label."""

    key: str
    value_labels: Mapping[int, str]
    output_prefix: str | None = None

    @property
    def resolved_output_prefix(self) -> str:
        return self.output_prefix or self.key


MetricSpec: TypeAlias = ScalarMetricSpec | CategoricalMetricSpec


def validate_metric_spec(spec: MetricSpec) -> None:
    """Validate one metric specification before collector construction."""

    if isinstance(spec, ScalarMetricSpec):
        validate_non_empty_string("key", spec.key)
        if not isinstance(spec.reducer, ScalarReducer):
            raise TypeError(f"reducer must be a ScalarReducer, got {type(spec.reducer)}.")
        if spec.output_name is not None:
            validate_non_empty_string("output_name", spec.output_name)
        return

    if not isinstance(spec, CategoricalMetricSpec):
        raise TypeError(
            "metric specification must be a ScalarMetricSpec or CategoricalMetricSpec, "
            f"got {type(spec)}."
        )

    validate_non_empty_string("key", spec.key)
    if spec.output_prefix is not None:
        validate_non_empty_string("output_prefix", spec.output_prefix)
    validate_mapping("value_labels", spec.value_labels)
    if not spec.value_labels:
        raise ValueError("value_labels must not be empty.")
    for value, label in spec.value_labels.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"Categorical values must be integers, got {value!r}.")
        validate_non_empty_string("category label", label)
    if len(set(spec.value_labels.values())) != len(spec.value_labels):
        raise ValueError("Categorical labels must be unique.")
