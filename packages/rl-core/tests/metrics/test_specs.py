from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

import rl_core
import rl_core.metrics as metrics
from rl_core.metrics import (
    CategoricalMetricSpec,
    MetricsCollector,
    ScalarMetricSpec,
    ScalarReducer,
    validate_metric_spec,
)


def test_metric_spec_public_api_is_exported() -> None:
    assert rl_core.ScalarMetricSpec is ScalarMetricSpec
    assert rl_core.CategoricalMetricSpec is CategoricalMetricSpec
    assert rl_core.ScalarReducer is ScalarReducer
    assert rl_core.validate_metric_spec is validate_metric_spec
    assert metrics.validate_metric_spec is validate_metric_spec
    assert "validate_metric_spec" in metrics.__all__


@pytest.mark.parametrize("spec_type", [ScalarMetricSpec, CategoricalMetricSpec])
def test_metric_specs_are_frozen_slotted_keyword_only_dataclasses(spec_type: type[object]) -> None:
    assert all(field.kw_only for field in fields(spec_type))

    if spec_type is ScalarMetricSpec:
        spec = ScalarMetricSpec(key="reward", reducer=ScalarReducer.MEAN)
        positional_args = ("reward", ScalarReducer.MEAN)
    else:
        spec = CategoricalMetricSpec(key="cause", value_labels={1: "TIMEOUT"})
        positional_args = ("cause", {1: "TIMEOUT"})

    assert not hasattr(spec, "__dict__")
    with pytest.raises(FrozenInstanceError):
        spec.key = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec_type(*positional_args)


def test_metric_specs_are_passive_values_and_validation_is_explicit() -> None:
    scalar = ScalarMetricSpec(key="", reducer="mean")  # type: ignore[arg-type]
    categorical = CategoricalMetricSpec(key="cause", value_labels={})

    assert scalar.key == ""
    assert categorical.value_labels == {}
    with pytest.raises(ValueError, match="key"):
        validate_metric_spec(scalar)
    with pytest.raises(ValueError, match="value_labels"):
        validate_metric_spec(categorical)


@pytest.mark.parametrize("key", [None, 1, False])
def test_scalar_spec_rejects_non_string_keys(key: object) -> None:
    with pytest.raises(TypeError, match="key"):
        validate_metric_spec(ScalarMetricSpec(key=key, reducer=ScalarReducer.MEAN))  # type: ignore[arg-type]


def test_scalar_spec_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="key"):
        validate_metric_spec(ScalarMetricSpec(key="", reducer=ScalarReducer.MEAN))


@pytest.mark.parametrize("reducer", ["mean", None, 1])
def test_scalar_spec_requires_scalar_reducer(reducer: object) -> None:
    with pytest.raises(TypeError, match="reducer"):
        validate_metric_spec(ScalarMetricSpec(key="reward", reducer=reducer))  # type: ignore[arg-type]


@pytest.mark.parametrize("output_name", [1, False])
def test_scalar_spec_rejects_non_string_output_name(output_name: object) -> None:
    with pytest.raises(TypeError, match="output_name"):
        validate_metric_spec(
            ScalarMetricSpec(
                key="reward",
                reducer=ScalarReducer.MEAN,
                output_name=output_name,  # type: ignore[arg-type]
            )
        )


def test_scalar_spec_rejects_empty_output_name() -> None:
    with pytest.raises(ValueError, match="output_name"):
        validate_metric_spec(
            ScalarMetricSpec(key="reward", reducer=ScalarReducer.MEAN, output_name="")
        )


def test_scalar_spec_resolves_default_and_explicit_output_names() -> None:
    assert (
        ScalarMetricSpec(key="reward", reducer=ScalarReducer.MEAN).resolved_output_name
        == "reward"
    )
    assert (
        ScalarMetricSpec(
            key="reward", reducer=ScalarReducer.MEAN, output_name="episode_reward"
        ).resolved_output_name
        == "episode_reward"
    )


@pytest.mark.parametrize("key", [None, 1, False])
def test_categorical_spec_rejects_non_string_keys(key: object) -> None:
    with pytest.raises(TypeError, match="key"):
        validate_metric_spec(CategoricalMetricSpec(key=key, value_labels={1: "END"}))  # type: ignore[arg-type]


def test_categorical_spec_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="key"):
        validate_metric_spec(CategoricalMetricSpec(key="", value_labels={1: "END"}))


@pytest.mark.parametrize("output_prefix", [1, False])
def test_categorical_spec_rejects_non_string_output_prefix(output_prefix: object) -> None:
    with pytest.raises(TypeError, match="output_prefix"):
        validate_metric_spec(
            CategoricalMetricSpec(
                key="cause",
                value_labels={1: "END"},
                output_prefix=output_prefix,  # type: ignore[arg-type]
            )
        )


def test_categorical_spec_rejects_empty_output_prefix() -> None:
    with pytest.raises(ValueError, match="output_prefix"):
        validate_metric_spec(
            CategoricalMetricSpec(key="cause", value_labels={1: "END"}, output_prefix="")
        )


@pytest.mark.parametrize("value_labels", [None, [], "labels"])
def test_categorical_spec_requires_mapping(value_labels: object) -> None:
    with pytest.raises(TypeError, match="value_labels"):
        validate_metric_spec(
            CategoricalMetricSpec(key="cause", value_labels=value_labels)  # type: ignore[arg-type]
        )


def test_categorical_spec_requires_non_empty_mapping() -> None:
    with pytest.raises(ValueError, match="value_labels"):
        validate_metric_spec(CategoricalMetricSpec(key="cause", value_labels={}))


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_categorical_values_must_be_non_boolean_integers(value: object) -> None:
    with pytest.raises(TypeError, match="Categorical values"):
        validate_metric_spec(CategoricalMetricSpec(key="cause", value_labels={value: "END"}))  # type: ignore[dict-item]


@pytest.mark.parametrize("label", [None, 1, False])
def test_category_labels_must_be_strings(label: object) -> None:
    with pytest.raises(TypeError, match="category label"):
        validate_metric_spec(CategoricalMetricSpec(key="cause", value_labels={1: label}))  # type: ignore[dict-item]


def test_category_labels_must_be_non_empty_and_unique() -> None:
    with pytest.raises(ValueError, match="category label"):
        validate_metric_spec(CategoricalMetricSpec(key="cause", value_labels={1: ""}))
    with pytest.raises(ValueError, match="unique"):
        validate_metric_spec(
            CategoricalMetricSpec(key="cause", value_labels={1: "END", 2: "END"})
        )


def test_categorical_spec_retains_caller_owned_value_labels_mapping() -> None:
    labels = {1: "TIMEOUT"}
    spec = CategoricalMetricSpec(key="cause", value_labels=labels)

    assert spec.value_labels is labels

    labels[2] = "COMPLETED"

    assert spec.value_labels == {1: "TIMEOUT", 2: "COMPLETED"}


def test_categorical_spec_resolves_default_and_explicit_output_prefixes() -> None:
    assert (
        CategoricalMetricSpec(key="cause", value_labels={1: "END"}).resolved_output_prefix
        == "cause"
    )
    assert (
        CategoricalMetricSpec(
            key="cause", value_labels={1: "END"}, output_prefix="end_cause"
        ).resolved_output_prefix
        == "end_cause"
    )


def test_validate_metric_spec_rejects_unknown_spec_type() -> None:
    with pytest.raises(TypeError, match="metric specification"):
        validate_metric_spec(object())  # type: ignore[arg-type]


def test_collector_validates_specs_at_consumption_boundary() -> None:
    invalid = ScalarMetricSpec(key="", reducer=ScalarReducer.MEAN)

    with pytest.raises(ValueError, match="key"):
        MetricsCollector(specs=[invalid])


def test_collector_rejects_duplicate_input_keys() -> None:
    with pytest.raises(ValueError, match="keys must be unique"):
        MetricsCollector(
            specs=[
                ScalarMetricSpec(key="reward", reducer=ScalarReducer.MEAN),
                ScalarMetricSpec(key="reward", reducer=ScalarReducer.SUM),
            ]
        )


@pytest.mark.parametrize(
    "specs",
    [
        [
            ScalarMetricSpec(
                key="reward", reducer=ScalarReducer.MEAN, output_name="same"
            ),
            ScalarMetricSpec(key="items", reducer=ScalarReducer.SUM, output_name="same"),
        ],
        [
            ScalarMetricSpec(
                key="reward", reducer=ScalarReducer.MEAN, output_name="cause_END"
            ),
            CategoricalMetricSpec(key="cause", value_labels={1: "END"}),
        ],
        [
            CategoricalMetricSpec(
                key="first", output_prefix="cause", value_labels={1: "END"}
            ),
            CategoricalMetricSpec(
                key="second", output_prefix="cause", value_labels={2: "END"}
            ),
        ],
    ],
)
def test_collector_rejects_all_duplicate_output_name_combinations(specs: list[object]) -> None:
    with pytest.raises(ValueError, match="duplicate output names"):
        MetricsCollector(specs=specs)  # type: ignore[arg-type]
