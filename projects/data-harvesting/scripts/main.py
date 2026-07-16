import argparse
import os

import mlflow
import yaml

from data_harvesting.train import train

from _paths import DEFAULT_PARAMS_PATH, DEFAULT_TRACKING_URI


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the data-harvesting policy.")
    parser.add_argument("-E", help="MLflow experiment name", dest="experiment_name")
    parser.add_argument("-R", help="MLflow run name", dest="run_id")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH), help="Path to params YAML")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="MLflow tracking URI",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name or "default")
    with open(args.params, "r", encoding="utf-8") as handle:
        config: dict = yaml.safe_load(handle)
    train(config, run_name=args.run_id)


if __name__ == "__main__":
    main()
