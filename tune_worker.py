"""Subprocess entrypoint for hyperparameter tuning.

This module is executed in a fresh Python process so any memory held by training
(PyTorch, environments, etc.) is released back to the OS when the process exits.

It reads the training config as JSON from stdin and prints a JSON object with the
numeric result to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mlflow

from data_harvesting.train import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune worker: run one train() call")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--result-path",
        default="",
        help="Optional path to write a JSON result payload to (used by the parent process).",
    )
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(args.tracking_uri)
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)

    config = json.loads(sys.stdin.read())
    result = train(config, run_name=args.run_name or None)

    payload = {"result": float(result)}

    if args.result_path:
        Path(args.result_path).write_text(json.dumps(payload) + "\n")

    # Still print a machine-readable payload for convenience/debugging.
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
