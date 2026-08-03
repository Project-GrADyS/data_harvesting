from pathlib import Path

import mlflow
import numpy as np

from sushigo.league import (
    RandomPolicy,
    TrainingRun,
    discover_training_runs,
    league_matchups,
    log_league_results,
    outcomes,
    run_league,
)


def _tracking_uri(path: Path) -> str:
    return path.resolve().as_uri()


def test_discovery_uses_all_completed_repetitions(tmp_path):
    uri = _tracking_uri(tmp_path / "mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("league-test")
    for repetition in (1, 2, 3):
        with mlflow.start_run():
            mlflow.set_tags(
                {
                    "sushigo.kind": "training",
                    "sushigo.checkpoint_schema": "1",
                    "sushigo.preset": "fixed_2p",
                    "sushigo.repetition": str(repetition),
                }
            )
            mlflow.log_text("placeholder", "policy/checkpoint.pt")
    discovered = discover_training_runs(
        tracking_uri=uri,
        experiment_name="league-test",
        presets=("fixed_2p",),
    )
    assert [run.repetition for run in discovered] == [1, 2, 3]


def test_matchups_exclude_repeated_families():
    runs = [
        TrainingRun("a1", "fixed_2p", 1),
        TrainingRun("a2", "fixed_2p", 2),
        TrainingRun("b1", "variable_2_4", 1),
        TrainingRun("random", "random", 1),
    ]
    matchups = league_matchups(runs, 2)
    assert matchups
    assert all(
        len({run.preset for run in matchup}) == 2 for matchup in matchups
    )
    assert not any(
        {run.run_id for run in matchup} == {"a1", "a2"}
        for matchup in matchups
    )


def test_matchups_exclude_fixed_policies_for_other_table_sizes():
    runs = [
        TrainingRun("fixed2", "fixed_2p", 1),
        TrainingRun("fixed3", "fixed_3p", 1),
        TrainingRun("variable", "variable_2_4", 1),
        TrainingRun("random", "random", 1),
    ]
    three_player = league_matchups(runs, 3)
    assert three_player
    assert all(
        "fixed_2p" not in {run.preset for run in matchup}
        for matchup in three_player
    )


def test_random_policy_is_legal_and_outcomes_use_pudding():
    policy = RandomPolicy(np.random.default_rng(5))
    observations = {
        "player_0": {"action_mask": np.array([0, 1, 0, 1], dtype=bool)}
    }
    assert policy.action(observations, 0) in {1, 3}
    assert outcomes([20, 20], [2, 1]) == ("win", "loss")
    assert outcomes([20, 20], [2, 2]) == ("tie", "tie")


class _FirstLegalPolicy:
    def __init__(self, spec):
        self.spec = spec

    def action(self, observations, seat):
        return int(
            np.flatnonzero(
                observations[f"player_{seat}"]["action_mask"]
            )[0]
        )


def test_league_is_seeded_and_logs_match_artifact(tmp_path):
    policies = [
        _FirstLegalPolicy(TrainingRun("fixed", "fixed_2p", 1)),
        _FirstLegalPolicy(TrainingRun("variable", "variable_2_4", 1)),
        _FirstLegalPolicy(TrainingRun("random", "random", 1)),
    ]
    first_rows, first_metrics = run_league(
        policies, players=2, games_per_matchup=1, seed=17
    )
    second_rows, second_metrics = run_league(
        policies, players=2, games_per_matchup=1, seed=17
    )
    assert first_rows == second_rows
    assert first_metrics == second_metrics

    uri = _tracking_uri(tmp_path / "league-mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("league-results")
    with mlflow.start_run() as active:
        log_league_results(first_rows, first_metrics)
        run_id = active.info.run_id
    artifacts = mlflow.MlflowClient().list_artifacts(run_id, "league")
    assert any(artifact.path == "league/matches.csv" for artifact in artifacts)
