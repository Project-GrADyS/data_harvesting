import argparse
import os

import mlflow
import numpy as np

from sushigo.league import (
    RANDOM_PRESET,
    TRAINING_PRESETS,
    RandomPolicy,
    discover_training_runs,
    league_matchups,
    load_run_policy,
    log_league_results,
    run_league,
)

from _paths import DEFAULT_TRACKING_URI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an MLflow-backed Sushi Go round-robin league."
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    parser.add_argument("--experiment", default="sushigo")
    parser.add_argument("--players", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--games-per-matchup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=TRAINING_PRESETS,
        default=list(TRAINING_PRESETS),
    )
    parser.add_argument("--repetitions", type=int, nargs="+")
    parser.add_argument(
        "--without-random-baseline", action="store_true"
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.games_per_matchup < 1:
        parser.error("--games-per-matchup must be at least 1")
    if args.repetitions and min(args.repetitions) < 1:
        parser.error("--repetitions must be positive")

    runs = discover_training_runs(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        presets=args.presets,
        repetitions=args.repetitions,
    )
    policies = [
        load_run_policy(
            run, tracking_uri=args.tracking_uri, device=args.device
        )
        for run in runs
    ]
    if not args.without_random_baseline:
        policies.append(RandomPolicy(np.random.default_rng(args.seed)))
    if not league_matchups([policy.spec for policy in policies], args.players):
        parser.error(
            "Not enough distinct completed policy families for this table size."
        )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"league-{args.players}p"):
        mlflow.set_tags(
            {
                "sushigo.kind": "league",
                "sushigo.players": str(args.players),
                "sushigo.seed": str(args.seed),
            }
        )
        mlflow.log_params(
            {
                "players": args.players,
                "games_per_matchup": args.games_per_matchup,
                "seed": args.seed,
                "source_run_ids": ",".join(run.run_id for run in runs),
                "random_baseline": not args.without_random_baseline,
            }
        )
        rows, metrics = run_league(
            policies,
            players=args.players,
            games_per_matchup=args.games_per_matchup,
            seed=args.seed,
        )
        log_league_results(rows, metrics)
        print(
            f"Logged {len(rows) // args.players} games and "
            f"{len(metrics)} standings metrics."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
