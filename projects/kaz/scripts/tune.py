from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from hyperopt import fmin, hp, space_eval, tpe
import mlflow
import yaml

from _paths import DEFAULT_PARAMS_PATH, DEFAULT_TRACKING_URI, PROJECT_ROOT


def _deep_update(target: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(child) for key, child in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune KAZ MADDPG hyperparameters.")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--experiment", default="kaz-maddpg-tuning")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "best_params.yaml"),
    )
    args = parser.parse_args(argv)

    with open(args.params, encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    base_config["training"]["total_timesteps"] = args.timesteps
    base_config["training"]["seed"] = args.seed
    base_config["checkpoint"]["enabled"] = False
    base_config["checkpoint"]["save_final_model"] = False

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    space = {
        "optimization": {
            "lr": hp.loguniform("lr", math.log(1e-5), math.log(3e-3)),
            "gamma": hp.uniform("gamma", 0.95, 0.999),
            "tau": hp.loguniform("tau", math.log(1e-3), math.log(5e-2)),
            "updates_per_batch": hp.choice("updates_per_batch", [1, 2, 4]),
        },
        "training": {
            "batch_size": hp.choice("batch_size", [128, 256, 512]),
            "exploration_epsilon_end": hp.choice(
                "exploration_epsilon_end", [0.02, 0.05, 0.1]
            ),
        },
    }

    def objective(sample: dict) -> float:
        run_config = deepcopy(base_config)
        _deep_update(run_config, _plain(sample))
        label = " ".join(
            [
                f"lr={run_config['optimization']['lr']:.3g}",
                f"gamma={run_config['optimization']['gamma']:.4f}",
                f"tau={run_config['optimization']['tau']:.3g}",
                f"batch={run_config['training']['batch_size']}",
            ]
        )
        handle, result_name = tempfile.mkstemp(prefix="kaz_tune_", suffix=".json")
        os.close(handle)
        result_path = Path(result_name)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("tune_worker.py")),
                    "--tracking-uri",
                    args.tracking_uri,
                    "--experiment",
                    args.experiment,
                    "--run-name",
                    label,
                    "--result-path",
                    str(result_path),
                ],
                input=json.dumps(run_config),
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Tuning worker exited with code {completed.returncode}.")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            return -float(result["score"])
        finally:
            result_path.unlink(missing_ok=True)

    encoded_best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=args.trials,
        rstate=__import__("numpy").random.default_rng(args.seed),
        show_progressbar=False,
    )
    best = _plain(space_eval(space, encoded_best))
    output = Path(args.output)
    output.write_text(yaml.safe_dump(best, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(best, sort_keys=False))
    print(f"Best parameters written to {output}")


if __name__ == "__main__":
    main()
