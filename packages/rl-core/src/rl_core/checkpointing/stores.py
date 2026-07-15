from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import tempfile

import torch

from validation_core import (
    validate_non_empty_string,
    validate_non_negative_integer,
    validate_positive_integer,
)

from .checkpoint import Checkpoint, _checkpoint_payload, load_checkpoint, validate_checkpoint


def _validate_filename_prefix(prefix: object) -> None:
    validate_non_empty_string("filename_prefix", prefix)
    if "/" in prefix or "\\" in prefix or prefix in {".", ".."}:
        raise ValueError("filename_prefix must not contain path components.")


def _checkpoint_filename(step: int, prefix: str) -> str:
    return f"{prefix}-step-{step:012d}.pt"


class LocalCheckpointStore:
    """Atomically persist checkpoints in a local directory with optional retention."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        keep_last: int | None = None,
        filename_prefix: str = "checkpoint",
    ) -> None:
        if keep_last is not None:
            validate_positive_integer("keep_last", keep_last)
        _validate_filename_prefix(filename_prefix)
        self.directory = Path(directory)
        self.keep_last = keep_last
        self.filename_prefix = filename_prefix
        self._filename_pattern = re.compile(
            rf"^{re.escape(filename_prefix)}-step-(\d+)\.pt$"
        )

    def save(self, checkpoint: Checkpoint) -> Path:
        validate_checkpoint(checkpoint)
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self._path_for_step(checkpoint.step)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.directory,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            torch.save(_checkpoint_payload(checkpoint), temporary_path)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._apply_retention()
        return destination

    def load(
        self,
        step: int,
        *,
        map_location: torch.device | str | None = "cpu",
    ) -> Checkpoint:
        validate_non_negative_integer("step", step)
        return load_checkpoint(self._path_for_step(step), map_location=map_location)

    def load_latest(
        self,
        *,
        map_location: torch.device | str | None = "cpu",
    ) -> Checkpoint:
        steps = self.list_steps()
        if not steps:
            raise FileNotFoundError(f"No checkpoints found in {self.directory}.")
        return self.load(steps[-1], map_location=map_location)

    def list_steps(self) -> tuple[int, ...]:
        if not self.directory.exists():
            return ()
        steps = [
            step
            for path in self.directory.iterdir()
            if path.is_file() and (step := self._step_from_name(path.name)) is not None
        ]
        return tuple(sorted(steps))

    def _path_for_step(self, step: int) -> Path:
        return self.directory / _checkpoint_filename(step, self.filename_prefix)

    def _step_from_name(self, name: str) -> int | None:
        match = self._filename_pattern.fullmatch(name)
        return int(match.group(1)) if match else None

    def _apply_retention(self) -> None:
        if self.keep_last is None:
            return
        steps = self.list_steps()
        for step in steps[:-self.keep_last]:
            self._path_for_step(step).unlink(missing_ok=True)


class MLflowCheckpointStore:
    """Persist checkpoints as MLflow run artifacts using the optional MLflow dependency."""

    def __init__(
        self,
        *,
        artifact_path: str = "checkpoints",
        run_id: str | None = None,
        tracking_uri: str | None = None,
        filename_prefix: str = "checkpoint",
    ) -> None:
        validate_non_empty_string("artifact_path", artifact_path)
        if not artifact_path.strip("/"):
            raise ValueError("artifact_path must identify a relative artifact path.")
        if PurePosixPath(artifact_path).is_absolute() or ".." in PurePosixPath(artifact_path).parts:
            raise ValueError("artifact_path must be a relative path without parent traversal.")
        if run_id is not None:
            validate_non_empty_string("run_id", run_id)
        if tracking_uri is not None:
            validate_non_empty_string("tracking_uri", tracking_uri)
        _validate_filename_prefix(filename_prefix)
        self.artifact_path = artifact_path.strip("/")
        self.run_id = run_id
        self.tracking_uri = tracking_uri
        self.filename_prefix = filename_prefix
        self._filename_pattern = re.compile(
            rf"^{re.escape(filename_prefix)}-step-(\d+)\.pt$"
        )

    def save(self, checkpoint: Checkpoint) -> str:
        validate_checkpoint(checkpoint)
        mlflow = self._import_mlflow()
        filename = _checkpoint_filename(checkpoint.step, self.filename_prefix)
        with tempfile.TemporaryDirectory() as temporary_directory:
            local_path = Path(temporary_directory) / filename
            torch.save(_checkpoint_payload(checkpoint), local_path)
            mlflow.log_artifact(
                str(local_path),
                artifact_path=self.artifact_path,
                run_id=self.run_id,
            )
        return str(PurePosixPath(self.artifact_path) / filename)

    def load(
        self,
        step: int,
        *,
        run_id: str | None = None,
        map_location: torch.device | str | None = "cpu",
    ) -> Checkpoint:
        validate_non_negative_integer("step", step)
        effective_run_id = self._resolve_run_id(run_id)
        filename = _checkpoint_filename(step, self.filename_prefix)
        return self._download_and_load(filename, effective_run_id, map_location)

    def load_latest(
        self,
        *,
        run_id: str | None = None,
        map_location: torch.device | str | None = "cpu",
    ) -> Checkpoint:
        mlflow = self._import_mlflow()
        effective_run_id = self._resolve_run_id(run_id)
        client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        artifacts = client.list_artifacts(effective_run_id, path=self.artifact_path)
        candidates: list[tuple[int, str]] = []
        for artifact in artifacts:
            if artifact.is_dir:
                continue
            name = PurePosixPath(artifact.path).name
            match = self._filename_pattern.fullmatch(name)
            if match:
                candidates.append((int(match.group(1)), name))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoints found for MLflow run {effective_run_id!r} under {self.artifact_path!r}."
            )
        _, filename = max(candidates)
        return self._download_and_load(filename, effective_run_id, map_location)

    def _download_and_load(
        self,
        filename: str,
        run_id: str,
        map_location: torch.device | str | None,
    ) -> Checkpoint:
        mlflow = self._import_mlflow()
        artifact_path = str(PurePosixPath(self.artifact_path) / filename)
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            tracking_uri=self.tracking_uri,
        )
        return load_checkpoint(local_path, map_location=map_location)

    def _resolve_run_id(self, run_id: str | None) -> str:
        if run_id is not None:
            validate_non_empty_string("run_id", run_id)
        effective_run_id = run_id or self.run_id
        if effective_run_id is not None:
            return effective_run_id
        mlflow = self._import_mlflow()
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("Loading an MLflow checkpoint requires a run_id or an active run.")
        return active_run.info.run_id

    @staticmethod
    def _import_mlflow():
        try:
            import mlflow
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "MLflowCheckpointStore requires the optional MLflow dependency. "
                "Install rl-core with the 'mlflow' extra."
            ) from error
        return mlflow
