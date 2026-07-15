from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from rl_core.checkpointing import Checkpoint, LocalCheckpointStore


@pytest.mark.parametrize("keep_last", [True, False, 1.0, "1"])
def test_local_store_rejects_wrong_keep_last_types(tmp_path: Path, keep_last: object) -> None:
    with pytest.raises(TypeError, match="keep_last"):
        LocalCheckpointStore(tmp_path, keep_last=keep_last)  # type: ignore[arg-type]


@pytest.mark.parametrize("keep_last", [0, -1])
def test_local_store_rejects_non_positive_keep_last(tmp_path: Path, keep_last: int) -> None:
    with pytest.raises(ValueError, match="keep_last"):
        LocalCheckpointStore(tmp_path, keep_last=keep_last)


@pytest.mark.parametrize("prefix", [None, 1, b"checkpoint"])
def test_local_store_rejects_wrong_filename_prefix_types(tmp_path: Path, prefix: object) -> None:
    with pytest.raises(TypeError, match="filename_prefix"):
        LocalCheckpointStore(tmp_path, filename_prefix=prefix)  # type: ignore[arg-type]


@pytest.mark.parametrize("prefix", ["", ".", "..", "nested/checkpoint", "nested\\checkpoint"])
def test_local_store_rejects_invalid_filename_prefix_values(tmp_path: Path, prefix: str) -> None:
    with pytest.raises(ValueError, match="filename_prefix"):
        LocalCheckpointStore(tmp_path, filename_prefix=prefix)


def test_local_store_round_trips_checkpoint_and_creates_directory(
    tmp_path: Path, checkpoint_factory
) -> None:
    store = LocalCheckpointStore(tmp_path / "nested" / "checkpoints")

    saved_path = store.save(checkpoint_factory(step=12, value=3.5))
    restored = store.load(12)

    assert saved_path.name == "checkpoint-step-000000000012.pt"
    assert saved_path.parent == store.directory
    assert restored.step == 12
    assert restored.metadata == {"algorithm": "test"}
    torch.testing.assert_close(restored.state["policy"]["weight"], torch.tensor(3.5))
    assert not list(saved_path.parent.glob("*.tmp"))


def test_local_store_uses_custom_prefix(tmp_path: Path, checkpoint_factory) -> None:
    store = LocalCheckpointStore(tmp_path, filename_prefix="agent")

    path = store.save(checkpoint_factory(step=2))

    assert path.name == "agent-step-000000000002.pt"
    assert store.list_steps() == (2,)


def test_local_store_overwrites_same_step(tmp_path: Path, checkpoint_factory) -> None:
    store = LocalCheckpointStore(tmp_path)

    first_path = store.save(checkpoint_factory(step=4, value=1.0))
    second_path = store.save(checkpoint_factory(step=4, value=2.0))

    assert first_path == second_path
    assert store.list_steps() == (4,)
    torch.testing.assert_close(store.load(4).state["policy"]["weight"], torch.tensor(2.0))


def test_local_store_lists_steps_numerically_and_ignores_unrelated_entries(
    tmp_path: Path, checkpoint_factory
) -> None:
    store = LocalCheckpointStore(tmp_path)
    for step in (20, 5, 12):
        store.save(checkpoint_factory(step=step))
    (tmp_path / "notes.txt").write_text("not a checkpoint")
    (tmp_path / "checkpoint-step-not-a-number.pt").write_text("not a checkpoint")
    (tmp_path / "checkpoint-step-000000000099.pt.dir").mkdir()

    assert store.list_steps() == (5, 12, 20)
    assert store.load_latest().step == 20


def test_local_store_returns_empty_steps_when_directory_does_not_exist(tmp_path: Path) -> None:
    assert LocalCheckpointStore(tmp_path / "missing").list_steps() == ()


def test_local_store_reports_missing_step_and_missing_latest(tmp_path: Path) -> None:
    store = LocalCheckpointStore(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load(1)
    with pytest.raises(FileNotFoundError, match="No checkpoints"):
        store.load_latest()


def test_local_store_retains_only_latest_numeric_steps(tmp_path: Path, checkpoint_factory) -> None:
    store = LocalCheckpointStore(tmp_path, keep_last=2)

    for step in (20, 5, 12):
        store.save(checkpoint_factory(step=step))

    assert store.list_steps() == (12, 20)


def test_local_store_cleans_temporary_file_when_serialization_fails(
    monkeypatch, tmp_path: Path, checkpoint_factory
) -> None:
    store = LocalCheckpointStore(tmp_path)

    def fail_save(*args, **kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(torch, "save", fail_save)

    with pytest.raises(RuntimeError, match="serialization failed"):
        store.save(checkpoint_factory(step=1))

    assert list(tmp_path.iterdir()) == []


def test_local_store_cleans_temporary_file_when_atomic_replace_fails(
    monkeypatch, tmp_path: Path, checkpoint_factory
) -> None:
    store = LocalCheckpointStore(tmp_path)

    def fail_replace(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.save(checkpoint_factory(step=1))

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("checkpoint", "exception"),
    [
        ({"step": 1}, TypeError),
        (Checkpoint(step="1", state={"policy": {}}), TypeError),  # type: ignore[arg-type]
        (Checkpoint(step=-1, state={"policy": {}}), ValueError),
        (Checkpoint(step=1, state={}), ValueError),
    ],
)
def test_local_store_validates_before_creating_directory(
    tmp_path: Path, checkpoint: object, exception: type[Exception]
) -> None:
    directory = tmp_path / "not-created"
    store = LocalCheckpointStore(directory)

    with pytest.raises(exception):
        store.save(checkpoint)  # type: ignore[arg-type]

    assert not directory.exists()


@pytest.mark.parametrize("step", [True, 1.0, "1"])
def test_local_store_load_rejects_wrong_step_types(tmp_path: Path, step: object) -> None:
    with pytest.raises(TypeError, match="step"):
        LocalCheckpointStore(tmp_path).load(step)  # type: ignore[arg-type]


def test_local_store_load_rejects_negative_step(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="step"):
        LocalCheckpointStore(tmp_path).load(-1)


def test_local_store_forwards_map_location(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, object]] = []

    def fake_load_checkpoint(path, *, map_location):
        calls.append((path, map_location))
        return Checkpoint(step=1, state={"policy": {}})

    monkeypatch.setattr("rl_core.checkpointing.stores.load_checkpoint", fake_load_checkpoint)
    device = torch.device("cpu")
    store = LocalCheckpointStore(tmp_path)

    store.load(1, map_location=device)

    assert calls == [(tmp_path / "checkpoint-step-000000000001.pt", device)]
