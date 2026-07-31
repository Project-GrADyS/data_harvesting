import argparse
import os

import mlflow

from sushigo.config import load_config, with_preset
from sushigo.environment.torchrl import resolve_player_counts
from sushigo.train import train

from _paths import DEFAULT_PARAMS_PATH, DEFAULT_TRACKING_URI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Sushi Go DQN policy.")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    parser.add_argument("--experiment", default="sushigo")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--preset",
        choices=(
            "fixed_2p",
            "fixed_3p",
            "fixed_4p",
            "variable_2_4",
            "variable_encoder_2_4",
        ),
    )
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repetition < 1:
        parser.error("--repetition must be at least 1")
    config = load_config(args.params)
    if args.preset:
        config = with_preset(config, args.preset)
    if args.device:
        config["training"]["device"] = args.device

    _, minimum, maximum, dense = resolve_player_counts(config["environment"])
    fixed = config["environment"].get("n_players")
    architecture = (
        "flex_encoder" if config["model"]["use_encoder"] else "mlp"
    )
    preset = args.preset or (
        f"fixed_{fixed}p"
        if fixed is not None
        else (
            f"variable_encoder_{minimum}_{maximum}"
            if config["model"]["use_encoder"]
            else f"variable_{minimum}_{maximum}"
        )
    )
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    train(
        config,
        run_name=args.run_name or f"{preset}-r{args.repetition}",
        tags={
            "sushigo.preset": preset,
            "sushigo.repetition": str(args.repetition),
            "sushigo.min_players": str(fixed or minimum),
            "sushigo.max_players": str(fixed or maximum),
            "sushigo.dense_players": str(dense),
            "sushigo.architecture": architecture,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
