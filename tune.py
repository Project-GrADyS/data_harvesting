import argparse
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy

import mlflow
from hyperopt import fmin, hp, tpe


parser = argparse.ArgumentParser()
parser.add_argument(
    "-E",
    type=str,
    required=False,
    help="MLflow experiment name (defaults to 'default')",
    dest="experiment_name",
)
args = parser.parse_args()

MLFLOW_TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

if __name__ == "__main__":
    import yaml

    with open("params.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
    
    experiment_name = args.experiment_name if args.experiment_name else "default"
    mlflow.set_experiment(experiment_name)
    
    space = {
        "optimization": {
            "num_optimizer_steps": hp.choice("num_optimizer_steps", [1, 5, 10, 20, 50]),
            "lr": hp.loguniform("lr", -10, -3),
            "grad_clip": hp.choice("grad_clip", [0, 0.5, 1.0, 5.0]),
        }
    }

    def _train_in_subprocess(run_config: dict, run_name: str | None) -> float:
        """Runs train() in a fresh Python process and returns its numeric result.

        This avoids memory accumulation in the parent process across many trials.
        """

        fd, result_path = tempfile.mkstemp(prefix="tune_result_", suffix=".json")
        os.close(fd)

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tune_worker",
                    "--tracking-uri",
                    MLFLOW_TRACKING_URI,
                    "--experiment-name",
                    experiment_name,
                    "--run-name",
                    run_name or "",
                    "--result-path",
                    result_path,
                ],
                input=json.dumps(run_config),
                text=True,
                # Inherit stdout/stderr so training progress is visible live.
                stdout=None,
                stderr=None,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            if completed.returncode != 0:
                raise RuntimeError(f"Training subprocess failed (exit_code={completed.returncode})")

            if not os.path.exists(result_path):
                raise RuntimeError(
                    "Training subprocess did not write a result file. "
                    "(Was it killed, or did it crash before finishing?)"
                )

            payload = json.loads(open(result_path, "r", encoding="utf-8").read())
            return float(payload["result"])
        finally:
            try:
                os.remove(result_path)
            except OSError:
                pass

    def tune(tune_args: dict):
        run_config = deepcopy(config)

        tuned_arg_leafs: list[tuple[str, object]] = []

        # tune_args contains overrides that should be merged into the base config
        for section, params in tune_args.items():
            if section not in run_config:
                run_config[section] = {}
            for key, value in params.items():
                run_config[section][key] = value
                tuned_arg_leafs.append((f"{section}.{key}", value))

        run_name = " ".join([f"{k}={v}" for k, v in tuned_arg_leafs])

        return -_train_in_subprocess(run_config, run_name=run_name)

    best = fmin(
        fn=tune,
        space=space,
        algo=tpe.suggest,
        max_evals=30,
        show_progressbar=False
    )
    print("Best hyperparameters found:")
    print(best)