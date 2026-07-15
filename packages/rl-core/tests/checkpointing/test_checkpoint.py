from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, fields
import pickle
from pathlib import Path

import pytest
import torch

import rl_core
from rl_core.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    Checkpoint,
    CheckpointManager,
    CheckpointStore,
    LocalCheckpointStore,
    MLflowCheckpointStore,
    load_checkpoint,
    validate_checkpoint,
)


def test_checkpoint_is_a_passive_frozen_slotted_keyword_only_value() -> None:
    state = {"policy": {}}
    metadata = {"algorithm": "test"}

    checkpoint = Checkpoint(step=-1, state=state, metadata=metadata)

    assert checkpoint.state is state
    assert checkpoint.metadata is metadata
    assert not hasattr(checkpoint, "__dict__")
    assert [(field.name, field.kw_only) for field in fields(Checkpoint)] == [
        ("step", True),
        ("state", True),
        ("metadata", True),
    ]
    with pytest.raises(FrozenInstanceError):
        checkpoint.step = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        Checkpoint(0, state)  # type: ignore[misc]


def test_checkpoint_mappings_remain_caller_owned() -> None:
    state = {"policy": {}}
    metadata = {"algorithm": "test"}
    checkpoint = Checkpoint(step=0, state=state, metadata=metadata)

    state["optimizer"] = {}
    metadata["algorithm"] = "changed"

    assert checkpoint.state == {"policy": {}, "optimizer": {}}
    assert checkpoint.metadata == {"algorithm": "changed"}


def test_checkpoint_metadata_default_is_not_shared() -> None:
    first = Checkpoint(step=0, state={"policy": {}})
    second = Checkpoint(step=1, state={"policy": {}})

    first.metadata["algorithm"] = "test"

    assert second.metadata == {}


def test_checkpoint_supports_standard_dataclass_and_pickle_tools(checkpoint_factory) -> None:
    checkpoint = checkpoint_factory(step=3, value=2.5)

    converted = asdict(checkpoint)
    restored = pickle.loads(pickle.dumps(checkpoint))

    assert converted["step"] == 3
    assert converted["metadata"] == {"algorithm": "test"}
    torch.testing.assert_close(converted["state"]["policy"]["weight"], torch.tensor(2.5))
    assert restored.step == checkpoint.step
    assert restored.metadata == checkpoint.metadata
    torch.testing.assert_close(
        restored.state["policy"]["weight"], checkpoint.state["policy"]["weight"]
    )


@pytest.mark.parametrize("step", [True, False, 1.0, "1", None])
def test_validate_checkpoint_rejects_wrong_step_types(step: object) -> None:
    with pytest.raises(TypeError, match="step"):
        validate_checkpoint(Checkpoint(step=step, state={"policy": {}}))  # type: ignore[arg-type]


def test_validate_checkpoint_rejects_negative_step() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        validate_checkpoint(Checkpoint(step=-1, state={"policy": {}}))


@pytest.mark.parametrize("state", [None, [], (), "state", 1])
def test_validate_checkpoint_rejects_non_mapping_state(state: object) -> None:
    with pytest.raises(TypeError, match="state"):
        validate_checkpoint(Checkpoint(step=0, state=state))  # type: ignore[arg-type]


@pytest.mark.parametrize("state", [{}, {"": {}}])
def test_validate_checkpoint_rejects_invalid_state_values(state: object) -> None:
    with pytest.raises(ValueError, match="state"):
        validate_checkpoint(Checkpoint(step=0, state=state))  # type: ignore[arg-type]


@pytest.mark.parametrize("state", [{1: {}}, {None: {}}])
def test_validate_checkpoint_rejects_wrong_state_key_types(state: object) -> None:
    with pytest.raises(TypeError, match="state"):
        validate_checkpoint(Checkpoint(step=0, state=state))  # type: ignore[arg-type]


@pytest.mark.parametrize("metadata", [None, [], (), "metadata", 1])
def test_validate_checkpoint_rejects_non_mapping_metadata(metadata: object) -> None:
    with pytest.raises(TypeError, match="metadata"):
        validate_checkpoint(
            Checkpoint(step=0, state={"policy": {}}, metadata=metadata)  # type: ignore[arg-type]
        )


def test_validate_checkpoint_rejects_empty_metadata_key() -> None:
    with pytest.raises(ValueError, match="metadata"):
        validate_checkpoint(
            Checkpoint(step=0, state={"policy": {}}, metadata={"": 1})
        )


@pytest.mark.parametrize("metadata", [{1: 1}, {None: 1}])
def test_validate_checkpoint_rejects_wrong_metadata_key_types(metadata: object) -> None:
    with pytest.raises(TypeError, match="metadata"):
        validate_checkpoint(
            Checkpoint(step=0, state={"policy": {}}, metadata=metadata)  # type: ignore[arg-type]
        )


def test_validate_checkpoint_accepts_empty_metadata_and_general_mappings() -> None:
    from collections import UserDict

    checkpoint = Checkpoint(
        step=0,
        state=UserDict({"policy": {}}),
        metadata=UserDict(),
    )

    assert validate_checkpoint(checkpoint) is None


def test_load_checkpoint_round_trips_versioned_payload(
    tmp_path: Path, checkpoint_factory
) -> None:
    path = LocalCheckpointStore(tmp_path).save(checkpoint_factory(step=12, value=3.5))

    payload = torch.load(path, weights_only=False)
    restored = load_checkpoint(path)

    assert payload == {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": 12,
        "state": restored.state,
        "metadata": {"algorithm": "test"},
    }
    assert restored.step == 12
    torch.testing.assert_close(restored.state["policy"]["weight"], torch.tensor(3.5))


@pytest.mark.parametrize("payload", [None, [], "checkpoint", 1])
def test_load_checkpoint_rejects_non_mapping_payload(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)

    with pytest.raises(TypeError, match="payload"):
        load_checkpoint(path)


@pytest.mark.parametrize("missing_key", ["format_version", "step", "state", "metadata"])
def test_load_checkpoint_rejects_each_missing_payload_key(
    tmp_path: Path, missing_key: str
) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": 1,
        "state": {"policy": {}},
        "metadata": {},
    }
    del payload[missing_key]
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="missing keys"):
        load_checkpoint(path)


def test_load_checkpoint_rejects_unsupported_format_version(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"format_version": 999, "step": 1, "state": {"policy": {}}, "metadata": {}},
        path,
    )

    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        load_checkpoint(path)


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("step", "1", TypeError),
        ("step", -1, ValueError),
        ("state", [], TypeError),
        ("state", {}, ValueError),
        ("metadata", [], TypeError),
        ("metadata", {"": 1}, ValueError),
    ],
)
def test_load_checkpoint_validates_loaded_checkpoint_fields(
    tmp_path: Path, field: str, value: object, exception: type[Exception]
) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": 1,
        "state": {"policy": {}},
        "metadata": {},
    }
    payload[field] = value
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)

    with pytest.raises(exception, match=field):
        load_checkpoint(path)


def test_load_checkpoint_forwards_map_location(monkeypatch, tmp_path: Path) -> None:
    calls: list[object] = []

    def fake_load(path, *, map_location, weights_only):
        calls.append((Path(path), map_location, weights_only))
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": 1,
            "state": {"policy": {}},
            "metadata": {},
        }

    monkeypatch.setattr(torch, "load", fake_load)
    device = torch.device("cpu")

    load_checkpoint(tmp_path / "checkpoint.pt", map_location=device)

    assert calls == [(tmp_path / "checkpoint.pt", device, False)]


def test_checkpointing_api_is_exported_from_feature_and_package_roots() -> None:
    assert rl_core.Checkpoint is Checkpoint
    assert rl_core.CheckpointManager is CheckpointManager
    assert rl_core.CheckpointStore is CheckpointStore
    assert rl_core.LocalCheckpointStore is LocalCheckpointStore
    assert rl_core.MLflowCheckpointStore is MLflowCheckpointStore
    assert rl_core.load_checkpoint is load_checkpoint
    assert rl_core.validate_checkpoint is validate_checkpoint
