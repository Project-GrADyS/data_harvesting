from __future__ import annotations

import ast
from collections import UserDict
from pathlib import Path

import pytest

import validation_core
from validation_core import (
    validate_bool,
    validate_callable,
    validate_mapping,
    validate_non_empty_string,
    validate_non_negative_integer,
    validate_positive_integer,
    validate_probability,
)


PUBLIC_API = {
    "validate_bool",
    "validate_callable",
    "validate_mapping",
    "validate_non_empty_string",
    "validate_non_negative_integer",
    "validate_positive_integer",
    "validate_probability",
}


def test_root_package_exports_exact_public_api() -> None:
    assert set(validation_core.__all__) == PUBLIC_API
    assert all(getattr(validation_core, name) is globals()[name] for name in PUBLIC_API)


def test_package_source_parses_as_python_3_11() -> None:
    source_root = Path(validation_core.__file__).parent

    for source_file in source_root.rglob("*.py"):
        ast.parse(
            source_file.read_text(encoding="utf-8"),
            filename=str(source_file),
            feature_version=(3, 11),
        )


@pytest.mark.parametrize("value", [1, 2, 10_000])
def test_validate_positive_integer_accepts_positive_integers(value: int) -> None:
    assert validate_positive_integer("count", value) is None


@pytest.mark.parametrize("value", [0, -1, -10_000])
def test_validate_positive_integer_rejects_out_of_range_integers(value: int) -> None:
    with pytest.raises(ValueError, match="count"):
        validate_positive_integer("count", value)


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None, object()])
def test_validate_positive_integer_rejects_wrong_runtime_types(value: object) -> None:
    with pytest.raises(TypeError, match="count"):
        validate_positive_integer("count", value)


@pytest.mark.parametrize("value", [0, 1, 10_000])
def test_validate_non_negative_integer_accepts_integers_at_or_above_zero(value: int) -> None:
    assert validate_non_negative_integer("step", value) is None


@pytest.mark.parametrize("value", [-1, -10_000])
def test_validate_non_negative_integer_rejects_negative_integers(value: int) -> None:
    with pytest.raises(ValueError, match="step"):
        validate_non_negative_integer("step", value)


@pytest.mark.parametrize("value", [True, False, 0.0, "0", None, object()])
def test_validate_non_negative_integer_rejects_wrong_runtime_types(value: object) -> None:
    with pytest.raises(TypeError, match="step"):
        validate_non_negative_integer("step", value)


@pytest.mark.parametrize("value", ["x", "multi word", " "])
def test_validate_non_empty_string_accepts_non_empty_strings(value: str) -> None:
    assert validate_non_empty_string("name", value) is None


def test_validate_non_empty_string_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="name"):
        validate_non_empty_string("name", "")


@pytest.mark.parametrize("value", [None, 1, True, b"bytes", object()])
def test_validate_non_empty_string_rejects_wrong_runtime_types(value: object) -> None:
    with pytest.raises(TypeError, match="name"):
        validate_non_empty_string("name", value)


@pytest.mark.parametrize("value", [0, 0.0, 0.5, 0.999_999])
def test_validate_probability_accepts_values_in_half_open_unit_interval(
    value: int | float,
) -> None:
    assert validate_probability("dropout", value) is None


@pytest.mark.parametrize(
    "value",
    [-1, -0.000_001, 1, 1.0, float("inf"), float("-inf"), float("nan")],
)
def test_validate_probability_rejects_values_outside_half_open_unit_interval(
    value: int | float,
) -> None:
    with pytest.raises(ValueError, match="dropout"):
        validate_probability("dropout", value)


@pytest.mark.parametrize("value", [True, False, "0.5", None, object()])
def test_validate_probability_rejects_wrong_runtime_types(value: object) -> None:
    with pytest.raises(TypeError, match="dropout"):
        validate_probability("dropout", value)


@pytest.mark.parametrize("value", [True, False])
def test_validate_bool_accepts_only_booleans(value: bool) -> None:
    assert validate_bool("enabled", value) is None


@pytest.mark.parametrize("value", [0, 1, 0.0, "true", None, object()])
def test_validate_bool_rejects_wrong_runtime_types(value: object) -> None:
    with pytest.raises(TypeError, match="enabled"):
        validate_bool("enabled", value)


@pytest.mark.parametrize("value", [{}, {"key": "value"}, UserDict({"key": "value"})])
def test_validate_mapping_accepts_mapping_implementations(value: object) -> None:
    assert validate_mapping("kwargs", value) is None


@pytest.mark.parametrize("value", [None, [], (), "mapping", object()])
def test_validate_mapping_rejects_wrong_runtime_types(value: object) -> None:
    with pytest.raises(TypeError, match="kwargs"):
        validate_mapping("kwargs", value)


class _CallableObject:
    def __call__(self) -> None:
        pass


@pytest.mark.parametrize("value", [lambda: None, len, _CallableObject, _CallableObject()])
def test_validate_callable_accepts_every_python_callable_form(value: object) -> None:
    assert validate_callable("factory", value) is None


@pytest.mark.parametrize("value", [None, 0, "callable", object()])
def test_validate_callable_rejects_non_callable_values(value: object) -> None:
    with pytest.raises(TypeError, match="factory"):
        validate_callable("factory", value)
