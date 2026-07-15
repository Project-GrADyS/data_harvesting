from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


ScheduledCallback = Callable[[int], object]


def _validate_non_negative_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")


def _validate_positive_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


@dataclass(frozen=True, slots=True)
class _Registration:
    name: str
    every: int
    callback: ScheduledCallback


class Scheduler:
    """Dispatch named callbacks at fixed step intervals."""

    def __init__(self) -> None:
        self._current_step = 0
        self._registrations: dict[str, _Registration] = {}

    @property
    def current_step(self) -> int:
        return self._current_step

    def register(self, name: str, *, every: int, callback: ScheduledCallback) -> None:
        """Register a synchronous callback under a unique name."""

        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string.")
        if name in self._registrations:
            raise ValueError(f"A callback named {name!r} is already registered.")
        _validate_positive_integer("every", every)
        if not callable(callback):
            raise TypeError("callback must be callable.")
        self._registrations[name] = _Registration(name, every, callback)

    def unregister(self, name: str) -> None:
        """Remove a registration by name."""

        try:
            del self._registrations[name]
        except KeyError:
            raise KeyError(f"No callback named {name!r} is registered.") from None

    def step(self, increment: int = 1) -> None:
        """Advance the clock and synchronously invoke callbacks whose interval was crossed.

        A callback is invoked at most once per call, even when ``increment`` crosses
        multiple occurrences of its interval. It receives the resulting current step.
        """

        _validate_positive_integer("increment", increment)
        previous_step = self._current_step
        self._current_step += increment

        registrations = tuple(self._registrations.values())
        for registration in registrations:
            if previous_step // registration.every < self._current_step // registration.every:
                registration.callback(self._current_step)

    def state_dict(self) -> dict[str, int]:
        """Return checkpointable scheduler state.

        Callback registrations are application code and are intentionally excluded.
        """

        return {"current_step": self._current_step}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore the scheduler clock while preserving current registrations."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping.")
        expected_keys = {"current_step"}
        actual_keys = set(state_dict)
        missing_keys = expected_keys - actual_keys
        unexpected_keys = actual_keys - expected_keys
        if missing_keys:
            raise ValueError(f"state_dict is missing keys: {sorted(missing_keys)}")
        if unexpected_keys:
            raise ValueError(f"state_dict has unexpected keys: {sorted(unexpected_keys)}")

        current_step = state_dict["current_step"]
        _validate_non_negative_integer("current_step", current_step)
        self._current_step = current_step


__all__ = ["ScheduledCallback", "Scheduler"]
