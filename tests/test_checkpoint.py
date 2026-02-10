import pytest
import torch
import tempfile
from pathlib import Path

from data_harvesting.checkpoint import save_checkpoint, load_checkpoint
from data_harvesting.metrics import EnvironmentMetricsCollector, LearningMetricsCollector


class MockAlgorithm:
    """Mock algorithm class for testing checkpointing."""
    
    def __init__(self):
        self.param = torch.tensor([1.0, 2.0, 3.0])
    
    def state_dict(self):
        return {"param": self.param}
    
    def load_state_dict(self, state_dict):
        self.param = state_dict["param"]


def test_save_and_load_checkpoint():
    """Test that checkpoints can be saved and loaded correctly."""
    device = torch.device("cpu")
    
    # Create mock objects
    algorithm = MockAlgorithm()
    metrics_logger = EnvironmentMetricsCollector(device)
    learning_logger = LearningMetricsCollector(device)
    
    # Set some state
    experience_steps = 1000
    iteration = 10
    metrics_logger.trajectories = torch.tensor(5.0)
    learning_logger.iterations = torch.tensor(10.0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
        
        # Save checkpoint
        save_checkpoint(
            checkpoint_path,
            algorithm,
            experience_steps,
            iteration,
            metrics_logger,
            learning_logger
        )
        
        assert checkpoint_path.exists(), "Checkpoint file should be created"
        
        # Create new instances to load into
        new_algorithm = MockAlgorithm()
        new_algorithm.param = torch.tensor([0.0, 0.0, 0.0])  # Different initial state
        new_metrics_logger = EnvironmentMetricsCollector(device)
        new_learning_logger = LearningMetricsCollector(device)
        
        # Load checkpoint
        loaded_state = load_checkpoint(
            checkpoint_path,
            new_algorithm,
            new_metrics_logger,
            new_learning_logger
        )
        
        # Verify loaded state
        assert loaded_state["experience_steps"] == experience_steps
        assert loaded_state["iteration"] == iteration
        assert torch.allclose(new_algorithm.param, algorithm.param)
        assert torch.allclose(new_metrics_logger.trajectories, metrics_logger.trajectories)
        assert torch.allclose(new_learning_logger.iterations, learning_logger.iterations)


def test_load_nonexistent_checkpoint():
    """Test that loading a nonexistent checkpoint raises FileNotFoundError."""
    device = torch.device("cpu")
    algorithm = MockAlgorithm()
    metrics_logger = EnvironmentMetricsCollector(device)
    learning_logger = LearningMetricsCollector(device)
    
    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            "/nonexistent/path/checkpoint.pt",
            algorithm,
            metrics_logger,
            learning_logger
        )


def test_metrics_state_dict_roundtrip():
    """Test that metrics state_dict can be saved and loaded correctly."""
    device = torch.device("cpu")
    
    # Environment metrics
    env_metrics = EnvironmentMetricsCollector(device)
    env_metrics.trajectories = torch.tensor(10.0)
    env_metrics.sum_avg_reward = torch.tensor(100.0)
    env_metrics.sum_max_reward = torch.tensor(50.0)
    
    state = env_metrics.state_dict()
    new_env_metrics = EnvironmentMetricsCollector(device)
    new_env_metrics.load_state_dict(state)
    
    assert torch.allclose(new_env_metrics.trajectories, env_metrics.trajectories)
    assert torch.allclose(new_env_metrics.sum_avg_reward, env_metrics.sum_avg_reward)
    assert torch.allclose(new_env_metrics.sum_max_reward, env_metrics.sum_max_reward)
    
    # Learning metrics
    learning_metrics = LearningMetricsCollector(device)
    learning_metrics.iterations = torch.tensor(20.0)
    learning_metrics.losses["loss_actor"] = torch.tensor(0.5)
    learning_metrics.losses["loss_value"] = torch.tensor(0.3)
    learning_metrics.start_time = 123.456
    
    state = learning_metrics.state_dict()
    new_learning_metrics = LearningMetricsCollector(device)
    new_learning_metrics.load_state_dict(state)
    
    assert torch.allclose(new_learning_metrics.iterations, learning_metrics.iterations)
    assert torch.allclose(new_learning_metrics.losses["loss_actor"], learning_metrics.losses["loss_actor"])
    assert torch.allclose(new_learning_metrics.losses["loss_value"], learning_metrics.losses["loss_value"])
    assert new_learning_metrics.start_time == learning_metrics.start_time
