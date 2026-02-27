from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol, TypedDict

import mlflow
import torch
from mlflow.artifacts import download_artifacts


CHECKPOINT_ARTIFACT_DIR = "checkpoints"


class Serializable(Protocol):
    def state_dict(self) -> Mapping[str, Any]:
        ...


class Loadable(Protocol):
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        ...


class CheckpointState(TypedDict):
    experience_steps: int
    iteration: int
    algorithm_state: Mapping[str, Any]
    metrics_logger_state: Mapping[str, Any]
    learning_logger_state: Mapping[str, Any]


def checkpoint_name_for_step(experience_steps: int) -> str:
    return f"checkpoint_step_{experience_steps}.pt"


def save_checkpoint(
    algorithm: Serializable,
    experience_steps: int,
    iteration: int,
    metrics_logger: Serializable,
    learning_logger: Serializable,
) -> str:
    """
    Save a training checkpoint as an MLflow artifact.
    
    Args:
        algorithm: The RL algorithm instance (MADDPGAlgorithm or MAPPOAlgorithm).
        experience_steps: Total experience steps completed so far.
        iteration: Current training iteration number.
        metrics_logger: EnvironmentMetricsCollector instance.
        learning_logger: LearningMetricsCollector instance.
    Returns:
        The checkpoint file name (e.g. checkpoint_step_100000.pt).
    """
    checkpoint: CheckpointState = {
        "experience_steps": experience_steps,
        "iteration": iteration,
        "algorithm_state": algorithm.state_dict(),
        "metrics_logger_state": metrics_logger.state_dict(),
        "learning_logger_state": learning_logger.state_dict(),
    }

    artifact_name = checkpoint_name_for_step(experience_steps)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / artifact_name
        torch.save(checkpoint, temp_path)
        mlflow.log_artifact(str(temp_path), artifact_path=CHECKPOINT_ARTIFACT_DIR)

    return artifact_name


def load_checkpoint(
    run_id: str,
    checkpoint_name: str,
    algorithm: Loadable,
    metrics_logger: Loadable,
    learning_logger: Loadable,
) -> dict[str, int]:
    """
    Load a training checkpoint from an MLflow run artifact.
    
    Args:
        run_id: MLflow run ID containing the checkpoint artifact.
        checkpoint_name: Checkpoint file name (e.g. checkpoint_step_100000.pt).
        algorithm: The RL algorithm instance to restore state into.
        metrics_logger: EnvironmentMetricsCollector instance to restore state into.
        learning_logger: LearningMetricsCollector instance to restore state into.
        
    Returns:
        Dictionary containing 'experience_steps' and 'iteration' from the checkpoint.
    """
    artifact_path = f"{CHECKPOINT_ARTIFACT_DIR}/{checkpoint_name}"
    local_path = Path(download_artifacts(run_id=run_id, artifact_path=artifact_path))

    checkpoint: CheckpointState = torch.load(local_path, weights_only=False)
    
    algorithm.load_state_dict(checkpoint["algorithm_state"])
    metrics_logger.load_state_dict(checkpoint["metrics_logger_state"])
    learning_logger.load_state_dict(checkpoint["learning_logger_state"])
    
    return {
        "experience_steps": checkpoint["experience_steps"],
        "iteration": checkpoint["iteration"],
    }
