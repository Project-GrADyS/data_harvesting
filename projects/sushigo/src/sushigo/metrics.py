"""Metric specifications and TensorDict extraction for Sushi Go."""

from __future__ import annotations

import torch
from rl_core import CategoricalMetricSpec, ScalarMetricSpec, ScalarReducer
from tensordict import TensorDictBase

from .environment.torchrl import GROUP, PLAYER_MASK_KEY


def training_metric_specs() -> tuple[ScalarMetricSpec, ...]:
    return (
        ScalarMetricSpec(key="loss", reducer=ScalarReducer.MEAN),
        ScalarMetricSpec(key="epsilon", reducer=ScalarReducer.MEAN),
        ScalarMetricSpec(key="frames_per_second", reducer=ScalarReducer.MEAN),
    )


def environment_metric_specs():
    return (
        ScalarMetricSpec(
            key="turn_reward",
            reducer=ScalarReducer.MEAN,
            output_name="mean_turn_reward",
        ),
        ScalarMetricSpec(
            key="episode_return",
            reducer=ScalarReducer.MEAN,
            output_name="mean_episode_return",
        ),
        ScalarMetricSpec(
            key="final_score",
            reducer=ScalarReducer.MEAN,
            output_name="mean_final_score",
        ),
        ScalarMetricSpec(
            key="winning_score",
            reducer=ScalarReducer.MEAN,
            output_name="mean_winning_score",
        ),
        ScalarMetricSpec(
            key="score_spread",
            reducer=ScalarReducer.MEAN,
            output_name="mean_score_spread",
        ),
        CategoricalMetricSpec(
            key="active_players",
            value_labels={2: "2p", 3: "3p", 4: "4p"},
            output_prefix="episodes",
        ),
    )


def extract_batch_metrics(batch: TensorDictBase) -> dict[str, torch.Tensor]:
    """Extract transition and terminal values without counting padded seats."""

    active = (
        batch.get(("next", *PLAYER_MASK_KEY)).squeeze(-1).to(torch.bool)
    )
    reward = batch.get(("next", GROUP, "reward")).squeeze(-1)
    values: dict[str, torch.Tensor] = {"turn_reward": reward[active]}

    done = (
        batch.get(("next", GROUP, "done")).squeeze(-1).to(torch.bool)
    )
    terminal_rows = done.any(dim=-1)
    if not bool(terminal_rows.any()):
        return values

    episode_return = batch.get(
        ("next", GROUP, "episode_reward")
    ).squeeze(-1)
    values["episode_return"] = episode_return[done & active]

    score = batch.get(
        ("next", GROUP, "observation", "episode_score")
    )
    values["final_score"] = score[done & active]

    for key in ("winning_score", "score_spread", "active_players"):
        tensor = batch.get(("next", GROUP, "observation", key))
        values[key] = tensor[terminal_rows, 0].reshape(-1)
    return values


def extract_terminal_metrics(
    terminal_transitions: TensorDictBase,
) -> dict[str, torch.Tensor]:
    """Evaluator adapter for transitions already selected as terminal."""

    active = terminal_transitions.get(
        ("next", *PLAYER_MASK_KEY)
    ).squeeze(-1).bool()
    episode_return = terminal_transitions.get(
        ("next", GROUP, "episode_reward")
    ).squeeze(-1)
    score = terminal_transitions.get(
        ("next", GROUP, "observation", "episode_score")
    )
    return {
        "episode_return": episode_return[active],
        "final_score": score[active],
        "winning_score": terminal_transitions.get(
            ("next", GROUP, "observation", "winning_score")
        )[..., 0, 0].reshape(-1),
        "score_spread": terminal_transitions.get(
            ("next", GROUP, "observation", "score_spread")
        )[..., 0, 0].reshape(-1),
        "active_players": terminal_transitions.get(
            ("next", GROUP, "observation", "active_players")
        )[..., 0, 0].reshape(-1),
    }
