from __future__ import annotations

import argparse
import os

import mlflow
import yaml

from kaz_training import train

from _paths import DEFAULT_PARAMS_PATH, DEFAULT_TRACKING_URI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a MADDPG policy for PettingZoo KAZ.")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--experiment", default="kaz-maddpg")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)

    with open(args.params, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.device is not None:
        config["training"]["device"] = args.device

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    score = train(config, run_name=args.run_name)
    print(f"Final mean team kills: {score:.3f}")


if __name__ == "__main__":
    main()
