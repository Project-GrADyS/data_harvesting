from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias


class ScalarReducer(StrEnum):
    """Supported reductions for scalar metric samples."""

    MEAN = "mean"
    SUM = "sum"


def _validate_name(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ScalarMetricSpec:
    """Declare a numeric metric reduced across every pushed sample."""

    key: str
    reducer: ScalarReducer
    output_name: str | None = None

    def __post_init__(self) -> None:
        _validate_name("key", self.key)
        if not isinstance(self.reducer, ScalarReducer):
            raise TypeError(f"reducer must be a ScalarReducer, got {type(self.reducer)}.")
        if self.output_name is not None:
            _validate_name("output_name", self.output_name)

    @property
    def resolved_output_name(self) -> str:
        return self.output_name or self.key


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoricalMetricSpec:
    """Declare a finite categorical metric expanded into one count per label."""

    key: str
    value_labels: Mapping[int, str]
    output_prefix: str | None = None

    def __post_init__(self) -> None:
        _validate_name("key", self.key)
        if self.output_prefix is not None:
            _validate_name("output_prefix", self.output_prefix)
        if not isinstance(self.value_labels, Mapping) or not self.value_labels:
            raise ValueError("value_labels must be a non-empty mapping.")

        copied_labels: dict[int, str] = {}
        for value, label in self.value_labels.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"Categorical values must be integers, got {value!r}.")
            _validate_name("category label", label)
            copied_labels[value] = label
        if len(set(copied_labels.values())) != len(copied_labels):
            raise ValueError("Categorical labels must be unique.")
        object.__setattr__(self, "value_labels", MappingProxyType(copied_labels))

    @property
    def resolved_output_prefix(self) -> str:
        return self.output_prefix or self.key


MetricSpec: TypeAlias = ScalarMetricSpec | CategoricalMetricSpec
