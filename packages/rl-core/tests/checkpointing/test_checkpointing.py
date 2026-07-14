from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest
import torch

import rl_core
from rl_core.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    Checkpoint,
    CheckpointManager,
    LocalCheckpointStore,
    MLflowCheckpointStore,
    load_checkpoint,
)


def make_checkpoint(step: int, value: float | None = None) -> Checkpoint:
    value = float(step) if value is None else value
    return Checkpoint(
        step=step,
        state={"policy": {"weight": torch.tensor(value)}},
        metadata={"algorithm": "test"},
    )


def test_checkpoint_validates_and_defensively_copies_mappings() -> None:
    state = {"policy": {"weight": torch.tensor(1.0)}}
    metadata = {"algorithm": "test"}
    checkpoint = Checkpoint(step=0, state=state, metadata=metadata)

    state["optimizer"] = {}
    metadata["algorithm"] = "changed"

    assert set(checkpoint.state) == {"policy"}
    assert checkpoint.metadata == {"algorithm": "test"}

    with pytest.raises(ValueError, match="non-negative"):
        Checkpoint(step=-1, state={"policy": {}})
    with pytest.raises(ValueError, match="must not be empty"):
        Checkpoint(step=0, state={})


def test_local_store_round_trip_and_explicit_load(tmp_path: Path) -> None:
    store = LocalCheckpointStore(tmp_path / "nested" / "checkpoints")
    checkpoint = make_checkpoint(12, 3.5)

    saved_path = store.save(checkpoint)
    restored = store.load(12)

    assert saved_path.name == "checkpoint-step-000000000012.pt"
    assert restored.step == 12
    assert restored.metadata == {"algorithm": "test"}
    torch.testing.assert_close(restored.state["policy"]["weight"], torch.tensor(3.5))
    assert not list(saved_path.parent.glob("*.tmp"))


def test_local_store_overwrites_same_step_atomically(tmp_path: Path) -> None:
    store = LocalCheckpointStore(tmp_path)

    first_path = store.save(make_checkpoint(4, 1.0))
    second_path = store.save(make_checkpoint(4, 2.0))

    assert first_path == second_path
    assert store.list_steps() == (4,)
    torch.testing.assert_close(store.load(4).state["policy"]["weight"], torch.tensor(2.0))


def test_local_store_retains_latest_steps(tmp_path: Path) -> None:
    store = LocalCheckpointStore(tmp_path, keep_last=2)

    for step in (20, 5, 12):
        store.save(make_checkpoint(step))

    assert store.list_steps() == (12, 20)
    assert store.load_latest().step == 20


def test_local_store_ignores_unrelated_files_and_reports_missing_latest(tmp_path: Path) -> None:
    store = LocalCheckpointStore(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "notes.txt").write_text("not a checkpoint")

    assert store.list_steps() == ()
    with pytest.raises(FileNotFoundError, match="No checkpoints"):
        store.load_latest()


def test_load_checkpoint_rejects_malformed_or_unsupported_payloads(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.pt"
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION}, malformed)
    with pytest.raises(ValueError, match="missing keys"):
        load_checkpoint(malformed)

    unsupported = tmp_path / "unsupported.pt"
    torch.save(
        {"format_version": 999, "step": 1, "state": {"policy": {}}, "metadata": {}},
        unsupported,
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        load_checkpoint(unsupported)


def test_local_store_validates_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_last"):
        LocalCheckpointStore(tmp_path, keep_last=0)
    with pytest.raises(ValueError, match="path components"):
        LocalCheckpointStore(tmp_path, filename_prefix="nested/checkpoint")


def test_checkpoint_manager_saves_to_stores_in_order() -> None:
    calls: list[tuple[str, int]] = []

    class Store:
        def __init__(self, name: str) -> None:
            self.name = name

        def save(self, checkpoint: Checkpoint) -> str:
            calls.append((self.name, checkpoint.step))
            return self.name

    manager = CheckpointManager([Store("local"), Store("remote")])

    assert manager.save(make_checkpoint(7)) == ("local", "remote")
    assert calls == [("local", 7), ("remote", 7)]


def test_checkpoint_manager_propagates_store_failure_and_stops() -> None:
    calls: list[str] = []

    class Store:
        def __init__(self, name: str, fails: bool = False) -> None:
            self.name = name
            self.fails = fails

        def save(self, checkpoint: Checkpoint) -> str:
            calls.append(self.name)
            if self.fails:
                raise RuntimeError("store unavailable")
            return self.name

    manager = CheckpointManager([Store("first"), Store("failing", True), Store("later")])

    with pytest.raises(RuntimeError, match="store unavailable"):
        manager.save(make_checkpoint(1))
    assert calls == ["first", "failing"]


def test_mlflow_store_saves_checkpoint_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str | None, Checkpoint]] = []

    def capture(local_path: str, artifact_path: str, run_id: str | None) -> None:
        calls.append((Path(local_path).name, artifact_path, run_id, load_checkpoint(local_path)))

    monkeypatch.setattr(mlflow, "log_artifact", capture)
    store = MLflowCheckpointStore(artifact_path="training/checkpoints", run_id="run-1")

    result = store.save(make_checkpoint(15, 2.5))

    assert result == "training/checkpoints/checkpoint-step-000000000015.pt"
    assert calls[0][:3] == ("checkpoint-step-000000000015.pt", "training/checkpoints", "run-1")
    assert calls[0][3].step == 15


def test_mlflow_store_loads_explicit_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local_store = LocalCheckpointStore(tmp_path)
    local_path = local_store.save(make_checkpoint(9, 4.0))
    calls: list[dict[str, object]] = []

    def download_artifacts(**kwargs) -> str:
        calls.append(kwargs)
        return str(local_path)

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download_artifacts)
    store = MLflowCheckpointStore(run_id="run-2", tracking_uri="http://tracking")

    restored = store.load(9)

    assert restored.step == 9
    assert calls == [
        {
            "run_id": "run-2",
            "artifact_path": "checkpoints/checkpoint-step-000000000009.pt",
            "tracking_uri": "http://tracking",
        }
    ]


def test_mlflow_store_loads_latest_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local_store = LocalCheckpointStore(tmp_path)
    latest_path = local_store.save(make_checkpoint(11))

    class Client:
        def list_artifacts(self, run_id: str, path: str):
            assert (run_id, path) == ("run-3", "checkpoints")
            return [
                SimpleNamespace(path="checkpoints/readme.txt", is_dir=False),
                SimpleNamespace(path="checkpoints/checkpoint-step-000000000002.pt", is_dir=False),
                SimpleNamespace(path="checkpoints/checkpoint-step-000000000011.pt", is_dir=False),
                SimpleNamespace(path="checkpoints/archive", is_dir=True),
            ]

    monkeypatch.setattr(mlflow, "MlflowClient", lambda tracking_uri=None: Client())
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", lambda **kwargs: str(latest_path))

    assert MLflowCheckpointStore(run_id="run-3").load_latest().step == 11


def test_mlflow_store_requires_run_for_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlflow, "active_run", lambda: None)

    with pytest.raises(RuntimeError, match="requires a run_id"):
        MLflowCheckpointStore().load(1)


def test_checkpointing_api_is_exported_from_package_root() -> None:
    assert rl_core.Checkpoint is Checkpoint
    assert rl_core.CheckpointManager is CheckpointManager
    assert rl_core.LocalCheckpointStore is LocalCheckpointStore
    assert rl_core.MLflowCheckpointStore is MLflowCheckpointStore
    assert rl_core.load_checkpoint is load_checkpoint
