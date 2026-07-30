from __future__ import annotations

import mlflow
import torch

from kaz_training.train import train


def test_short_training_loop_logs_and_evaluates(kaz_config, tmp_path) -> None:
    kaz_config["metrics"]["log_every_n_steps"] = 8
    kaz_config["evaluation"].update(
        {"enabled": True, "eval_every_n_steps": 8, "num_episodes": 1, "seed": 100}
    )
    mlflow.set_tracking_uri(tmp_path.resolve().as_uri())
    mlflow.set_experiment("kaz-smoke-test")

    score = train(kaz_config, run_name="smoke")

    assert torch.isfinite(torch.tensor(score))
    runs = mlflow.search_runs(experiment_names=["kaz-smoke-test"])
    assert len(runs) == 1
    assert "metrics.eval/final_team_kills_mean" in runs.columns
