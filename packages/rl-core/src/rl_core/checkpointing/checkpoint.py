from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import torch


CHECKPOINT_FORMAT_VERSION = 1


def _validate_step(step: object) -> None:
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}.")


def _copy_string_keyed_mapping(name: str, value: object, *, allow_empty: bool) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    copied = dict(value)
    if not allow_empty and not copied:
        raise ValueError(f"{name} must not be empty.")
    invalid_keys = [key for key in copied if not isinstance(key, str) or not key]
    if invalid_keys:
        raise ValueError(f"{name} keys must be non-empty strings, got {invalid_keys!r}.")
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True, kw_only=True)
class Checkpoint:
    """A versioned snapshot assembled by a project at one logical training step."""

    step: int
    state: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_step(self.step)
        object.__setattr__(self, "state", _copy_string_keyed_mapping("state", self.state, allow_empty=False))
        object.__setattr__(self, "metadata", _copy_string_keyed_mapping("metadata", self.metadata, allow_empty=True))


class CheckpointStore(Protocol):
    """A destination capable of persisting a checkpoint."""

    def save(self, checkpoint: Checkpoint) -> object: ...


def _checkpoint_payload(checkpoint: Checkpoint) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": checkpoint.step,
        "state": dict(checkpoint.state),
        "metadata": dict(checkpoint.metadata),
    }


def _checkpoint_from_payload(payload: object) -> Checkpoint:
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping.")
    required_keys = {"format_version", "step", "state", "metadata"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise ValueError(f"Checkpoint payload is missing keys: {sorted(missing_keys)}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version {payload['format_version']!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}."
        )
    return Checkpoint(
        step=payload["step"],
        state=payload["state"],
        metadata=payload["metadata"],
    )


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: torch.device | str | None = "cpu",
) -> Checkpoint:
    """Load and validate a trusted checkpoint file.

    PyTorch checkpoint files can execute arbitrary pickle payloads. Only load files from trusted sources.
    """

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    return _checkpoint_from_payload(payload)


class CheckpointManager:
    """Fan caller-created checkpoints out to one or more stores in order."""

    def __init__(self, stores: Iterable[CheckpointStore]) -> None:
        self._stores = tuple(stores)
        if not self._stores:
            raise ValueError("stores must contain at least one checkpoint store.")
        if not all(callable(getattr(store, "save", None)) for store in self._stores):
            raise TypeError("Every checkpoint store must provide a callable save(checkpoint) method.")

    def save(self, checkpoint: Checkpoint) -> tuple[object, ...]:
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError(f"checkpoint must be a Checkpoint, got {type(checkpoint)}.")
        return tuple(store.save(checkpoint) for store in self._stores)
