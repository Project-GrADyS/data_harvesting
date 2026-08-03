import argparse
import os

from sushigo.league import TRAINING_PRESETS
from sushigo.league_training import build_jobs, pending_jobs, run_jobs

from _paths import (
    DEFAULT_PARAMS_PATH,
    DEFAULT_TRACKING_URI,
    TRAIN_SCRIPT,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the Sushi Go league training matrix."
    )
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    parser.add_argument("--experiment", default="sushigo")
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=TRAINING_PRESETS,
        default=list(TRAINING_PRESETS),
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.parallelism < 1:
        parser.error("--parallelism must be at least 1")

    jobs = build_jobs(args.presets, args.repetitions)
    pending, completed = pending_jobs(
        jobs,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
    )
    for job in completed:
        print(f"SKIP {job.label}: completed MLflow run exists")

    def command(job):
        return job.command(
            train_script=TRAIN_SCRIPT,
            params_path=args.params,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
            device=args.device,
        )

    for job in pending:
        print(f"RUN  {job.label}: {' '.join(command(job))}")
    if args.dry_run or not pending:
        return 0
    return 0 if run_jobs(
        pending,
        parallelism=args.parallelism,
        command_factory=command,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
