from __future__ import annotations

import pytest

import rl_core
from rl_core.metrics import CategoricalMetricSpec, MetricsCollector, ScalarMetricSpec, ScalarReducer


def test_public_api_is_exported_from_package_root() -> None:
    assert rl_core.MetricsCollector is MetricsCollector
    assert rl_core.ScalarMetricSpec is ScalarMetricSpec
    assert rl_core.CategoricalMetricSpec is CategoricalMetricSpec
    assert rl_core.ScalarReducer is ScalarReducer


def test_specs_validate_names_reducers_and_labels() -> None:
    with pytest.raises(ValueError, match="key"):
        ScalarMetricSpec(key="", reducer=ScalarReducer.MEAN)
    with pytest.raises(TypeError, match="ScalarReducer"):
        ScalarMetricSpec(key="reward", reducer="mean")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty mapping"):
        CategoricalMetricSpec(key="cause", value_labels={})
    with pytest.raises(ValueError, match="unique"):
        CategoricalMetricSpec(key="cause", value_labels={1: "END", 2: "END"})


def test_collector_rejects_duplicate_input_and_output_names() -> None:
    with pytest.raises(ValueError, match="keys must be unique"):
        MetricsCollector(
            specs=[
                ScalarMetricSpec(key="reward", reducer=ScalarReducer.MEAN),
                ScalarMetricSpec(key="reward", reducer=ScalarReducer.SUM),
            ]
        )

    with pytest.raises(ValueError, match="duplicate output names"):
        MetricsCollector(
            specs=[
                ScalarMetricSpec(key="reward", reducer=ScalarReducer.MEAN, output_name="same"),
                ScalarMetricSpec(key="items", reducer=ScalarReducer.SUM, output_name="same"),
            ]
        )


def test_categorical_labels_are_defensively_copied() -> None:
    labels = {1: "TIMEOUT"}
    spec = CategoricalMetricSpec(key="cause", value_labels=labels)

    labels[2] = "COMPLETED"

    assert dict(spec.value_labels) == {1: "TIMEOUT"}
