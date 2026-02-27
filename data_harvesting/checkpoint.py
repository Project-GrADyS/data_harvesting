from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import torch
from mlflow import MlflowClient


_STEP_RE = re.compile(r"checkpoint_step_(\d+)\.pt$")


def save_checkpoint(
    algorithm: Any,
    experience_steps: int,
    iteration: int,
    metrics_logger: Any,
    learning_logger: Any,
    *,
    artifact_dir: str = "checkpoints",
) -> str:
    """
    Save a training checkpoint as an MLflow artifact.
    
    Args:
        algorithm: The RL algorithm instance (MADDPGAlgorithm or MAPPOAlgorithm).
        experience_steps: Total experience steps completed so far.
        iteration: Current training iteration number.
        metrics_logger: EnvironmentMetricsCollector instance.
        learning_logger: LearningMetricsCollector instance.
        artifact_dir: Artifact directory inside the run.

    Returns:
        The artifact path of the logged checkpoint file.
    """
    checkpoint = {
        "experience_steps": experience_steps,
        "iteration": iteration,
        "algorithm_state": algorithm.state_dict(),
        "metrics_logger_state": metrics_logger.state_dict(),
        "learning_logger_state": learning_logger.state_dict(),
    }

    artifact_name = f"checkpoint_step_{experience_steps}.pt"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / artifact_name
        torch.save(checkpoint, temp_path)
        mlflow.log_artifact(str(temp_path), artifact_path=artifact_dir)

    return f"{artifact_dir}/{artifact_name}"


def latest_checkpoint_artifact_path(
    *,
    run_id: str,
    artifact_dir: str = "checkpoints",
    tracking_uri: str | None = None,
) -> str | None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, path=artifact_dir)

    latest_step = -1
    latest_path: str | None = None
    for artifact in artifacts:
        if artifact.is_dir:
            continue
        match = _STEP_RE.search(Path(artifact.path).name)
        if not match:
            continue
        step = int(match.group(1))
        if step > latest_step:
            latest_step = step
            latest_path = artifact.path

    return latest_path


def load_checkpoint(
    checkpoint_path: str | Path | None,
    algorithm: Any,
    metrics_logger: Any,
    learning_logger: Any,
    *,
    run_id: str | None = None,
    artifact_dir: str = "checkpoints",
    tracking_uri: str | None = None,
) -> dict[str, int]:
    """
    Load a training checkpoint from local file or MLflow artifact.
    
    Args:
        checkpoint_path: Path to local checkpoint file, MLflow artifact path, or None.
        algorithm: The RL algorithm instance to restore state into.
        metrics_logger: EnvironmentMetricsCollector instance to restore state into.
        learning_logger: LearningMetricsCollector instance to restore state into.
        run_id: Optional MLflow run ID for artifact resolution.
        artifact_dir: Artifact directory where checkpoints are stored.
        tracking_uri: Optional MLflow tracking URI.
        
    Returns:
        Dictionary containing 'experience_steps' and 'iteration' from the checkpoint.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    artifact_path: str | None = None
    local_path: Path | None = None

    if checkpoint_path is None:
        if not run_id:
            active_run = mlflow.active_run()
            if active_run is None:
                raise ValueError("run_id is required when checkpoint_path is None")
            run_id = active_run.info.run_id
        artifact_path = latest_checkpoint_artifact_path(
            run_id=run_id,
            artifact_dir=artifact_dir,
            tracking_uri=tracking_uri,
        )
        if artifact_path is None:
            raise FileNotFoundError(f"No checkpoint artifacts found under '{artifact_dir}' for run '{run_id}'")
    else:
        path_text = str(checkpoint_path)
        local_candidate = Path(path_text)

        if local_candidate.exists():
            local_path = local_candidate
        elif path_text.startswith("runs:/"):
            local_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=path_text))
        else:
            artifact_path = path_text

    if local_path is None:
        if not run_id:
            active_run = mlflow.active_run()
            if active_run is None:
                raise ValueError("run_id is required when loading a checkpoint artifact outside an active MLflow run")
            run_id = active_run.info.run_id
        local_path = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path))

    checkpoint = torch.load(local_path, weights_only=False)
    
    algorithm.load_state_dict(checkpoint["algorithm_state"])
    metrics_logger.load_state_dict(checkpoint["metrics_logger_state"])
    learning_logger.load_state_dict(checkpoint["learning_logger_state"])
    
    return {
        "experience_steps": checkpoint["experience_steps"],
        "iteration": checkpoint["iteration"],
    }
