from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mlflow

from kaz_training import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one isolated KAZ tuning trial.")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--result-path", required=True)
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    config = json.loads(sys.stdin.read())
    score = train(config, run_name=args.run_name)
    Path(args.result_path).write_text(json.dumps({"score": score}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
