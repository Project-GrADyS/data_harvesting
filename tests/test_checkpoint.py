import pytest
import torch
import tempfile
import mlflow

from data_harvesting.checkpoint import load_checkpoint, save_checkpoint


def _init_test_mlflow(tmpdir: str) -> None:
    mlflow.set_tracking_uri(f"file:{tmpdir}")
    mlflow.set_experiment("checkpoint-tests")


class MockAlgorithm:
    """Mock algorithm class for testing checkpointing."""
    
    def __init__(self):
        self.param = torch.tensor([1.0, 2.0, 3.0])
    
    def state_dict(self):
        return {"param": self.param}
    
    def load_state_dict(self, state_dict):
        self.param = state_dict["param"]


class MockMetricsLogger:
    """Mock metrics logger for testing checkpointing."""
    
    def __init__(self):
        self.counter = torch.tensor(0.0)
        self.value = torch.tensor(10.0)
    
    def state_dict(self):
        return {
            "counter": self.counter,
            "value": self.value,
        }
    
    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]
        self.value = state_dict["value"]


def test_save_and_load_checkpoint():
    """Test that checkpoints can be saved and loaded correctly."""
    # Create mock objects
    algorithm = MockAlgorithm()
    metrics_logger = MockMetricsLogger()
    learning_logger = MockMetricsLogger()
    
    # Set some state
    experience_steps = 1000
    iteration = 10
    metrics_logger.counter = torch.tensor(5.0)
    learning_logger.counter = torch.tensor(10.0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        _init_test_mlflow(tmpdir)
        with mlflow.start_run() as run:
            # Save checkpoint
            checkpoint_name = save_checkpoint(
                algorithm,
                experience_steps,
                iteration,
                metrics_logger,
                learning_logger,
            )
        
            # Create new instances to load into
            new_algorithm = MockAlgorithm()
            new_algorithm.param = torch.tensor([0.0, 0.0, 0.0])  # Different initial state
            new_metrics_logger = MockMetricsLogger()
            new_learning_logger = MockMetricsLogger()
        
            # Load checkpoint
            loaded_state = load_checkpoint(
                run.info.run_id,
                checkpoint_name,
                new_algorithm,
                new_metrics_logger,
                new_learning_logger,
            )

            # Verify loaded state
            assert loaded_state["experience_steps"] == experience_steps
            assert loaded_state["iteration"] == iteration
            assert torch.allclose(new_algorithm.param, algorithm.param)
            assert torch.allclose(new_metrics_logger.counter, metrics_logger.counter)
            assert torch.allclose(new_learning_logger.counter, learning_logger.counter)


def test_load_nonexistent_checkpoint():
    """Test that loading a nonexistent checkpoint raises FileNotFoundError."""
    algorithm = MockAlgorithm()
    metrics_logger = MockMetricsLogger()
    learning_logger = MockMetricsLogger()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        _init_test_mlflow(tmpdir)
        with mlflow.start_run() as run:
            with pytest.raises(Exception):
                load_checkpoint(
                    run.info.run_id,
                    "checkpoint_step_12345.pt",
                    algorithm,
                    metrics_logger,
                    learning_logger
                )


def test_load_checkpoint_with_explicit_checkpoint_name():
    algorithm = MockAlgorithm()
    metrics_logger = MockMetricsLogger()
    learning_logger = MockMetricsLogger()

    with tempfile.TemporaryDirectory() as tmpdir:
        _init_test_mlflow(tmpdir)
        with mlflow.start_run() as run:
            save_checkpoint(algorithm, 100, 1, metrics_logger, learning_logger)

            loaded_state = load_checkpoint(
                run.info.run_id,
                "checkpoint_step_100.pt",
                algorithm,
                metrics_logger,
                learning_logger,
            )

            assert loaded_state["experience_steps"] == 100
            assert loaded_state["iteration"] == 1


def test_checkpoint_file_structure():
    """Test that checkpoint file contains the expected structure."""
    algorithm = MockAlgorithm()
    metrics_logger = MockMetricsLogger()
    learning_logger = MockMetricsLogger()

    experience_steps = 2000
    iteration = 20

    with tempfile.TemporaryDirectory() as tmpdir:
        _init_test_mlflow(tmpdir)
        with mlflow.start_run(run_name="checkpoint-structure") as run:
            checkpoint_name = save_checkpoint(
                algorithm,
                experience_steps,
                iteration,
                metrics_logger,
                learning_logger,
            )

            local_checkpoint_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path=f"checkpoints/{checkpoint_name}",
            )
            checkpoint = torch.load(local_checkpoint_path, weights_only=False)

            assert "experience_steps" in checkpoint
            assert "iteration" in checkpoint
            assert "algorithm_state" in checkpoint
            assert "metrics_logger_state" in checkpoint
            assert "learning_logger_state" in checkpoint

            assert checkpoint["experience_steps"] == experience_steps
            assert checkpoint["iteration"] == iteration
