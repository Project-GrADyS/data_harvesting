import torch
from pathlib import Path
from typing import Dict, Any, Optional


def save_checkpoint(
    checkpoint_path: str | Path,
    algorithm: Any,
    experience_steps: int,
    iteration: int,
    metrics_logger: Any,
    learning_logger: Any,
) -> None:
    """
    Save a training checkpoint to disk.
    
    Args:
        checkpoint_path: Path where the checkpoint will be saved.
        algorithm: The RL algorithm instance (MADDPGAlgorithm or MAPPOAlgorithm).
        experience_steps: Total experience steps completed so far.
        iteration: Current training iteration number.
        metrics_logger: EnvironmentMetricsCollector instance.
        learning_logger: LearningMetricsCollector instance.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "experience_steps": experience_steps,
        "iteration": iteration,
        "algorithm_state": algorithm.state_dict(),
        "metrics_logger_state": metrics_logger.state_dict(),
        "learning_logger_state": learning_logger.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    algorithm: Any,
    metrics_logger: Any,
    learning_logger: Any,
) -> Dict[str, int]:
    """
    Load a training checkpoint from disk.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        algorithm: The RL algorithm instance to restore state into.
        metrics_logger: EnvironmentMetricsCollector instance to restore state into.
        learning_logger: LearningMetricsCollector instance to restore state into.
        
    Returns:
        Dictionary containing 'experience_steps' and 'iteration' from the checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    algorithm.load_state_dict(checkpoint["algorithm_state"])
    metrics_logger.load_state_dict(checkpoint["metrics_logger_state"])
    learning_logger.load_state_dict(checkpoint["learning_logger_state"])
    
    return {
        "experience_steps": checkpoint["experience_steps"],
        "iteration": checkpoint["iteration"],
    }
