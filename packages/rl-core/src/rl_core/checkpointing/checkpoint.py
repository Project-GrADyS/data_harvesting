from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Protocol

import torch
from validation_core import (
    validate_mapping,
    validate_non_empty_string,
    validate_non_negative_integer,
    validate_positive_integer,
)


CHECKPOINT_FORMAT_VERSION = 1


def _validate_string_keyed_mapping(name: str, value: object, *, allow_empty: bool) -> None:
    validate_mapping(name, value)
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty.")
    for key in value:
        validate_non_empty_string(f"{name} key", key)


@dataclass(frozen=True, slots=True, kw_only=True)
class Checkpoint:
    """A versioned snapshot assembled by a project at one logical training step."""

    step: int
    state: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)



def validate_checkpoint(checkpoint: Checkpoint) -> None:
    """Validate a checkpoint before persistence or restoration."""

    if not isinstance(checkpoint, Checkpoint):
        raise TypeError(f"checkpoint must be a Checkpoint, got {type(checkpoint)}.")
    validate_non_negative_integer("step", checkpoint.step)
    _validate_string_keyed_mapping("state", checkpoint.state, allow_empty=False)
    _validate_string_keyed_mapping("metadata", checkpoint.metadata, allow_empty=True)


class CheckpointStore(Protocol):
    """A destination capable of persisting a checkpoint."""

    def save(self, checkpoint: Checkpoint) -> object: ...


def _checkpoint_payload(checkpoint: Checkpoint) -> dict[str, Any]:
    validate_checkpoint(checkpoint)
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": checkpoint.step,
        "state": checkpoint.state,
        "metadata": checkpoint.metadata,
    }


def _checkpoint_from_payload(payload: object) -> Checkpoint:
    validate_mapping("checkpoint payload", payload)
    required_keys = {"format_version", "step", "state", "metadata"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise ValueError(f"Checkpoint payload is missing keys: {sorted(missing_keys)}")
    format_version = payload["format_version"]
    validate_positive_integer("format_version", format_version)
    if format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version {format_version!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}."
        )
    checkpoint = Checkpoint(
        step=payload["step"],
        state=payload["state"],
        metadata=payload["metadata"],
    )
    validate_checkpoint(checkpoint)
    return checkpoint


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
        try:
            self._stores = tuple(stores)
        except TypeError as error:
            raise TypeError("stores must be an iterable of checkpoint stores.") from error
        if not self._stores:
            raise ValueError("stores must contain at least one checkpoint store.")
        if not all(callable(getattr(store, "save", None)) for store in self._stores):
            raise TypeError("Every checkpoint store must provide a callable save(checkpoint) method.")

    def save(self, checkpoint: Checkpoint) -> tuple[object, ...]:
        validate_checkpoint(checkpoint)
        return tuple(store.save(checkpoint) for store in self._stores)
