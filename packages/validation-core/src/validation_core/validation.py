from __future__ import annotations

from collections.abc import Mapping
from math import isfinite


def _validate_integer_type(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    return value


def validate_positive_integer(name: str, value: object) -> None:
    integer = _validate_integer_type(name, value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive, got {integer!r}.")


def validate_non_negative_integer(name: str, value: object) -> None:
    integer = _validate_integer_type(name, value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative, got {integer!r}.")


def validate_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}.")
    if not value:
        raise ValueError(f"{name} must not be empty.")


def validate_probability(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    if not 0.0 <= value < 1.0 or not isfinite(value):
        raise ValueError(f"{name} must be in the range [0.0, 1.0), got {value!r}.")


def validate_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}.")


def validate_mapping(name: str, value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")


def validate_callable(name: str, value: object) -> None:
    if not callable(value):
        raise TypeError(f"{name} must be callable, got {type(value).__name__}.")
