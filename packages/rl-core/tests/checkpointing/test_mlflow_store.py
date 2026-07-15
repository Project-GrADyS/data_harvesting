from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_core.checkpointing import Checkpoint, LocalCheckpointStore, MLflowCheckpointStore


class FakeMLflow:
    def __init__(self) -> None:
        self.logged: list[tuple[str, str, str | None]] = []
        self.downloads: list[dict[str, object]] = []
        self.artifacts = self
        self.client = None
        self.active = None

    def log_artifact(self, local_path: str, *, artifact_path: str, run_id: str | None) -> None:
        self.logged.append((local_path, artifact_path, run_id))

    def download_artifacts(self, **kwargs) -> str:
        self.downloads.append(kwargs)
        return self.download_path

    def MlflowClient(self, tracking_uri=None):
        self.client_tracking_uri = tracking_uri
        return self.client

    def active_run(self):
        return self.active


@pytest.mark.parametrize("artifact_path", [None, 1, b"checkpoints"])
def test_mlflow_store_rejects_wrong_artifact_path_types(artifact_path: object) -> None:
    with pytest.raises(TypeError, match="artifact_path"):
        MLflowCheckpointStore(artifact_path=artifact_path)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "artifact_path",
    ["", "/", "/absolute", "../checkpoints", "training/../checkpoints"],
)
def test_mlflow_store_rejects_invalid_artifact_paths(artifact_path: str) -> None:
    with pytest.raises(ValueError, match="artifact_path"):
        MLflowCheckpointStore(artifact_path=artifact_path)


def test_mlflow_store_normalizes_artifact_path_edges() -> None:
    store = MLflowCheckpointStore(artifact_path="training/checkpoints/")

    assert store.artifact_path == "training/checkpoints"


@pytest.mark.parametrize("run_id", [1, b"run", []])
def test_mlflow_store_rejects_wrong_run_id_types(run_id: object) -> None:
    with pytest.raises(TypeError, match="run_id"):
        MLflowCheckpointStore(run_id=run_id)  # type: ignore[arg-type]


def test_mlflow_store_rejects_empty_run_id() -> None:
    with pytest.raises(ValueError, match="run_id"):
        MLflowCheckpointStore(run_id="")


def test_mlflow_store_validates_filename_prefix() -> None:
    with pytest.raises(TypeError, match="filename_prefix"):
        MLflowCheckpointStore(filename_prefix=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="filename_prefix"):
        MLflowCheckpointStore(filename_prefix="nested/checkpoint")


def test_mlflow_store_saves_artifact_and_removes_temporary_file(
    monkeypatch, checkpoint_factory
) -> None:
    fake = FakeMLflow()
    loaded: list[Checkpoint] = []

    def capture(local_path: str, *, artifact_path: str, run_id: str | None) -> None:
        assert Path(local_path).exists()
        loaded.append(torch.load(local_path, weights_only=False))
        fake.logged.append((local_path, artifact_path, run_id))

    fake.log_artifact = capture
    store = MLflowCheckpointStore(
        artifact_path="training/checkpoints", run_id="run-1", filename_prefix="agent"
    )
    monkeypatch.setattr(store, "_import_mlflow", lambda: fake)

    result = store.save(checkpoint_factory(step=15, value=2.5))

    local_path, artifact_path, run_id = fake.logged[0]
    assert result == "training/checkpoints/agent-step-000000000015.pt"
    assert Path(local_path).name == "agent-step-000000000015.pt"
    assert not Path(local_path).exists()
    assert (artifact_path, run_id) == ("training/checkpoints", "run-1")
    assert loaded[0]["step"] == 15


@pytest.mark.parametrize(
    ("checkpoint", "exception"),
    [
        ({"step": 1}, TypeError),
        (Checkpoint(step="1", state={"policy": {}}), TypeError),  # type: ignore[arg-type]
        (Checkpoint(step=-1, state={"policy": {}}), ValueError),
        (Checkpoint(step=1, state={}), ValueError),
    ],
)
def test_mlflow_store_validates_before_importing_mlflow(
    monkeypatch, checkpoint: object, exception: type[Exception]
) -> None:
    store = MLflowCheckpointStore()
    monkeypatch.setattr(
        store,
        "_import_mlflow",
        lambda: pytest.fail("MLflow should not be imported for an invalid checkpoint"),
    )

    with pytest.raises(exception):
        store.save(checkpoint)  # type: ignore[arg-type]


def test_mlflow_store_loads_explicit_checkpoint(
    monkeypatch, tmp_path: Path, checkpoint_factory
) -> None:
    local_path = LocalCheckpointStore(tmp_path).save(checkpoint_factory(step=9, value=4.0))
    fake = FakeMLflow()
    fake.download_path = str(local_path)
    store = MLflowCheckpointStore(run_id="run-2", tracking_uri="http://tracking")
    monkeypatch.setattr(store, "_import_mlflow", lambda: fake)

    restored = store.load(9)

    assert restored.step == 9
    assert fake.downloads == [
        {
            "run_id": "run-2",
            "artifact_path": "checkpoints/checkpoint-step-000000000009.pt",
            "tracking_uri": "http://tracking",
        }
    ]


def test_mlflow_store_explicit_load_run_id_overrides_configured_run(
    monkeypatch, tmp_path: Path, checkpoint_factory
) -> None:
    local_path = LocalCheckpointStore(tmp_path).save(checkpoint_factory(step=3))
    fake = FakeMLflow()
    fake.download_path = str(local_path)
    store = MLflowCheckpointStore(run_id="configured")
    monkeypatch.setattr(store, "_import_mlflow", lambda: fake)

    store.load(3, run_id="override")

    assert fake.downloads[0]["run_id"] == "override"


@pytest.mark.parametrize("run_id", [1, b"run", []])
def test_mlflow_store_load_rejects_wrong_override_run_id_types(
    monkeypatch, run_id: object
) -> None:
    store = MLflowCheckpointStore(run_id="configured")
    monkeypatch.setattr(
        store,
        "_import_mlflow",
        lambda: pytest.fail("MLflow should not be used for an invalid run_id"),
    )

    with pytest.raises(TypeError, match="run_id"):
        store.load(1, run_id=run_id)  # type: ignore[arg-type]


def test_mlflow_store_load_rejects_empty_override_run_id(monkeypatch) -> None:
    store = MLflowCheckpointStore(run_id="configured")
    monkeypatch.setattr(
        store,
        "_import_mlflow",
        lambda: pytest.fail("MLflow should not be used for an invalid run_id"),
    )

    with pytest.raises(ValueError, match="run_id"):
        store.load(1, run_id="")


def test_mlflow_store_uses_active_run_when_no_run_is_given(
    monkeypatch, tmp_path: Path, checkpoint_factory
) -> None:
    local_path = LocalCheckpointStore(tmp_path).save(checkpoint_factory(step=3))
    fake = FakeMLflow()
    fake.download_path = str(local_path)
    fake.active = SimpleNamespace(info=SimpleNamespace(run_id="active-run"))
    store = MLflowCheckpointStore()
    monkeypatch.setattr(store, "_import_mlflow", lambda: fake)

    store.load(3)

    assert fake.downloads[0]["run_id"] == "active-run"


def test_mlflow_store_requires_run_for_loading(monkeypatch) -> None:
    fake = FakeMLflow()
    store = MLflowCheckpointStore()
    monkeypatch.setattr(store, "_import_mlflow", lambda: fake)

    with pytest.raises(RuntimeError, match="requires a run_id"):
        store.load(1)


def test_mlflow_store_loads_latest_numeric_checkpoint(
    monkeypatch, tmp_path: Path, checkpoint_factory
) -> None:
    latest_path = LocalCheckpointStore(tmp_path).save(checkpoint_factory(step=11))
    fake = FakeMLflow()
    fake.download_path = str(latest_path)
    fake.client = SimpleNamespace(
        list_artifacts=lambda run_id, path: [
            SimpleNamespace(path="checkpoints/readme.txt", is_dir=False),
            SimpleNamespace(path="checkpoints/checkpoint-step-000000000002.pt", is_dir=False),
            SimpleNamespace(path="checkpoints/checkpoint-step-000000000011.pt", is_dir=False),
            SimpleNamespace(path="checkpoints/archive", is_dir=True),
        ]
    )
    store = MLflowCheckpointStore(run_id="run-3", tracking_uri="tracking")
    monkeypatch.setattr(store, "_import_mlflow", lambda: fake)

    restored = store.load_latest(map_location=torch.device("cpu"))

    assert restored.step == 11
    assert fake.client_tracking_uri == "tracking"
    assert fake.downloads[0]["artifact_path"].endswith("000000000011.pt")


def test_mlflow_store_forwards_map_location_when_loading(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    fake = FakeMLflow()
    fake.download_path = "/tmp/downloaded.pt"
    store = MLflowCheckpointStore(run_id="run")
    monkeypatch.setattr(store, "_import_mlflow", lambda: fake)
    monkeypatch.setattr(
        "rl_core.checkpointing.stores.load_checkpoint",
        lambda path, *, map_location: calls.append((path, map_location)),
    )
    device = torch.device("cpu")

    store.load(1, map_location=device)

    assert calls == [("/tmp/downloaded.pt", device)]


def test_mlflow_store_reports_when_latest_has_no_candidates(monkeypatch) -> None:
    fake = FakeMLflow()
    fake.client = SimpleNamespace(
        list_artifacts=lambda run_id, path: [
            SimpleNamespace(path="checkpoints/readme.txt", is_dir=False),
            SimpleNamespace(path="checkpoints/archive", is_dir=True),
        ]
    )
    store = MLflowCheckpointStore(run_id="run-3")
    monkeypatch.setattr(store, "_import_mlflow", lambda: fake)

    with pytest.raises(FileNotFoundError, match="No checkpoints"):
        store.load_latest()


def test_mlflow_store_reports_missing_optional_dependency(monkeypatch) -> None:
    real_import = builtins.__import__

    def import_without_mlflow(name, *args, **kwargs):
        if name == "mlflow":
            raise ModuleNotFoundError("No module named 'mlflow'", name="mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_mlflow)

    with pytest.raises(ModuleNotFoundError, match="mlflow.*extra"):
        MLflowCheckpointStore._import_mlflow()
