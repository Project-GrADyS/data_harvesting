from pathlib import Path

import mlflow

from sushigo.config import load_config, with_preset
from sushigo.train import train

PARAMS = Path(__file__).parents[1] / "params.yaml"


def test_short_training_run_logs_checkpoint(tmp_path):
    config = with_preset(load_config(PARAMS), "fixed_2p")
    config["training"]["total_frames"] = 64
    config["training"]["device"] = "cpu"
    config["collector"]["frames_per_batch"] = 32
    config["collector"]["num_workers"] = 1
    config["replay_buffer"]["batch_size"] = 8
    config["replay_buffer"]["capacity"] = 128
    config["optimization"]["updates_per_batch"] = 1
    config["metrics"]["log_every_frames"] = 32
    config["checkpoint"]["every_frames"] = 0
    config["evaluation"]["enabled"] = True
    config["evaluation"]["every_frames"] = 64
    config["evaluation"]["num_episodes"] = 1

    mlflow.set_tracking_uri((tmp_path / "mlruns").resolve().as_uri())
    mlflow.set_experiment("smoke")
    score = train(
        config,
        run_name="smoke",
        tags={"sushigo.preset": "fixed_2p", "sushigo.repetition": "1"},
    )
    assert isinstance(score, float)
    run = mlflow.search_runs(experiment_names=["smoke"]).iloc[0]
    client = mlflow.MlflowClient()
    artifacts = client.list_artifacts(run.run_id, "policy")
    assert any(artifact.path == "policy/checkpoint.pt" for artifact in artifacts)
