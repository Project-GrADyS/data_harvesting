"""MLflow-backed Sushi Go policy discovery and round-robin evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, Sequence

import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from tensordict import TensorDict

from .environment import (
    ACTION_KEY,
    GROUP,
    MASK_KEY,
    OBS_COMPONENTS,
    PLAYER_MASK_KEY,
    SushiGoParallelEnv,
)
from .policy import CHECKPOINT_SCHEMA_VERSION, load_checkpoint_policy

RANDOM_PRESET = "random"
TRAINING_PRESETS = (
    "fixed_2p",
    "fixed_3p",
    "fixed_4p",
    "variable_2_4",
    "variable_encoder_2_4",
)
CSV_FIELDS = (
    "matchup",
    "game",
    "competitor",
    "run_id",
    "preset",
    "repetition",
    "seat",
    "points",
    "pudding",
    "outcome",
    "opponents",
)


@dataclass(frozen=True, slots=True)
class TrainingRun:
    run_id: str
    preset: str
    repetition: int

    @property
    def label(self) -> str:
        return f"{self.preset}/repetition_{self.repetition}"


class LeaguePolicy(Protocol):
    spec: TrainingRun

    def action(self, observations, seat: int) -> int: ...


def discover_training_runs(
    *,
    tracking_uri: str,
    experiment_name: str,
    presets: Sequence[str] = TRAINING_PRESETS,
    repetitions: Sequence[int] | None = None,
) -> list[TrainingRun]:
    """Discover every finished, schema-compatible training repetition."""

    unknown = set(presets).difference(TRAINING_PRESETS)
    if unknown:
        raise ValueError(f"Unknown presets: {sorted(unknown)}")
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    allowed_repetitions = None if repetitions is None else set(repetitions)
    discovered: dict[tuple[str, int], TrainingRun] = {}
    for run in client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
    ):
        tags = run.data.tags
        if run.info.status != "FINISHED":
            continue
        if tags.get("sushigo.kind") != "training":
            continue
        if tags.get("sushigo.checkpoint_schema") != str(
            CHECKPOINT_SCHEMA_VERSION
        ):
            continue
        preset = tags.get("sushigo.preset")
        repetition_text = tags.get("sushigo.repetition")
        if preset not in presets or repetition_text is None:
            continue
        repetition = int(repetition_text)
        if (
            allowed_repetitions is not None
            and repetition not in allowed_repetitions
        ):
            continue
        if not any(
            artifact.path == "policy/checkpoint.pt"
            for artifact in client.list_artifacts(run.info.run_id, "policy")
        ):
            continue
        discovered.setdefault(
            (preset, repetition),
            TrainingRun(
                run_id=run.info.run_id,
                preset=preset,
                repetition=repetition,
            ),
        )
    return sorted(
        discovered.values(), key=lambda run: (run.preset, run.repetition)
    )


def league_matchups(
    runs: Sequence[TrainingRun], players: int
) -> list[tuple[TrainingRun, ...]]:
    """Build compatible tables without repeated model families."""

    return [
        matchup
        for matchup in combinations(runs, players)
        if len({run.preset for run in matchup}) == players
        and all(_supports_players(run, players) for run in matchup)
    ]


def _supports_players(run: TrainingRun, players: int) -> bool:
    if run.preset in {RANDOM_PRESET, "variable_2_4", "variable_encoder_2_4"}:
        return True
    return run.preset == f"fixed_{players}p"


def _native_tensordict(
    observations,
    environment_config: dict[str, Any],
    device: torch.device,
) -> TensorDict:
    fixed = environment_config.get("n_players")
    dense_players = int(fixed or environment_config["max_n_players"])
    history_len = int(
        environment_config.get("history_len") or dense_players - 1
    )
    native = [
        observations[f"player_{seat}"] for seat in range(dense_players)
    ]
    tensordict = TensorDict({}, batch_size=[], device=device)
    for component in OBS_COMPONENTS:
        values = []
        for observation in native:
            value = observation[component]
            if component in {"hand_history", "opponent_tableaus"}:
                value = value[:history_len]
            values.append(value)
        tensordict.set(
            (GROUP, "observation", component),
            torch.as_tensor(np.stack(values), dtype=torch.float32, device=device),
        )
    for key in ("hand_history_mask", "opponent_tableaus_mask"):
        values = np.stack(
            [observation[key][:history_len] for observation in native]
        )
        tensordict.set(
            (GROUP, "observation", key),
            torch.as_tensor(values, dtype=torch.bool, device=device),
        )
    tensordict.set(
        MASK_KEY,
        torch.as_tensor(
            np.stack([observation["action_mask"] for observation in native]),
            dtype=torch.bool,
            device=device,
        ),
    )
    tensordict.set(
        PLAYER_MASK_KEY,
        torch.as_tensor(
            np.asarray([observation["player_mask"] for observation in native])[
                :, None
            ],
            dtype=torch.bool,
            device=device,
        ),
    )
    return tensordict


class LoadedPolicy:
    def __init__(
        self,
        spec: TrainingRun,
        policy: torch.nn.Module,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.spec = spec
        self.policy = policy
        self.config = config
        self.device = device

    def action(self, observations, seat: int) -> int:
        tensordict = _native_tensordict(
            observations, self.config["environment"], self.device
        )
        with torch.inference_mode():
            self.policy(tensordict)
        return int(tensordict.get(ACTION_KEY)[seat].item())


class RandomPolicy:
    def __init__(self, rng: np.random.Generator) -> None:
        self.spec = TrainingRun(
            run_id=RANDOM_PRESET, preset=RANDOM_PRESET, repetition=1
        )
        self.rng = rng

    def action(self, observations, seat: int) -> int:
        legal = np.flatnonzero(
            observations[f"player_{seat}"]["action_mask"]
        )
        return int(self.rng.choice(legal))


def load_run_policy(
    run: TrainingRun,
    *,
    tracking_uri: str,
    device: torch.device | str = "cpu",
) -> LoadedPolicy:
    path = mlflow.artifacts.download_artifacts(
        run_id=run.run_id,
        artifact_path="policy/checkpoint.pt",
        tracking_uri=tracking_uri,
    )
    policy, config = load_checkpoint_policy(path, device=device)
    return LoadedPolicy(run, policy, config, torch.device(device))


def outcomes(
    points: Sequence[int], puddings: Sequence[int]
) -> tuple[str, ...]:
    best_points = max(points)
    leaders = [
        index for index, value in enumerate(points) if value == best_points
    ]
    if len(leaders) > 1:
        best_pudding = max(puddings[index] for index in leaders)
        leaders = [
            index
            for index in leaders
            if puddings[index] == best_pudding
        ]
    if len(leaders) > 1:
        return tuple(
            "tie" if index in leaders else "loss"
            for index in range(len(points))
        )
    return tuple(
        "win" if index == leaders[0] else "loss"
        for index in range(len(points))
    )


def play_game(
    policies: Sequence[LeaguePolicy],
    *,
    players: int,
    rng: np.random.Generator,
    seed: int,
):
    order = rng.permutation(len(policies))
    seated = [policies[int(index)] for index in order]
    environment = SushiGoParallelEnv(
        n_players=None,
        min_n_players=2,
        max_n_players=4,
        reward_scale=1.0,
    )
    observations, _ = environment.reset(
        seed=seed, options={"n_players": players}
    )
    while environment.agents:
        actions = {
            f"player_{seat}": policy.action(observations, seat)
            for seat, policy in enumerate(seated)
        }
        observations, _, _, _, _ = environment.step(actions)
    points = [
        int(round(environment.episode_scores[seat]))
        for seat in range(players)
    ]
    puddings = [
        int(environment.pudding_total[seat]) for seat in range(players)
    ]
    environment.close()
    return seated, points, puddings, outcomes(points, puddings)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    labels = sorted({str(row["competitor"]) for row in rows})
    for label in labels:
        selected = [row for row in rows if row["competitor"] == label]
        prefix = label.replace("/", "_")
        result[f"{prefix}/win_rate"] = sum(
            row["outcome"] == "win" for row in selected
        ) / len(selected)
        result[f"{prefix}/tie_rate"] = sum(
            row["outcome"] == "tie" for row in selected
        ) / len(selected)
        result[f"{prefix}/mean_points"] = float(
            np.mean([row["points"] for row in selected])
        )
        result[f"{prefix}/mean_pudding"] = float(
            np.mean([row["pudding"] for row in selected])
        )
    return result


def run_league(
    policies: Sequence[LeaguePolicy],
    *,
    players: int,
    games_per_matchup: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    specs = [policy.spec for policy in policies]
    matchups = league_matchups(specs, players)
    policies_by_run = {policy.spec.run_id: policy for policy in policies}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    game_index = 0
    for matchup_index, matchup in enumerate(matchups, start=1):
        table = [policies_by_run[spec.run_id] for spec in matchup]
        for _ in range(games_per_matchup):
            game_index += 1
            seated, points, puddings, game_outcomes = play_game(
                table,
                players=players,
                rng=rng,
                seed=seed + game_index,
            )
            for seat, policy in enumerate(seated):
                rows.append(
                    {
                        "matchup": matchup_index,
                        "game": game_index,
                        "competitor": policy.spec.label,
                        "run_id": policy.spec.run_id,
                        "preset": policy.spec.preset,
                        "repetition": policy.spec.repetition,
                        "seat": seat,
                        "points": points[seat],
                        "pudding": puddings[seat],
                        "outcome": game_outcomes[seat],
                        "opponents": "|".join(
                            opponent.spec.label
                            for index, opponent in enumerate(seated)
                            if index != seat
                        ),
                    }
                )
    return rows, _aggregate(rows)


def log_league_results(
    rows: list[dict[str, Any]], metrics: dict[str, float]
) -> None:
    with TemporaryDirectory(prefix="sushigo-league-") as directory:
        path = Path(directory) / "matches.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        mlflow.log_artifact(path, artifact_path="league")
    mlflow.log_metrics(metrics)
