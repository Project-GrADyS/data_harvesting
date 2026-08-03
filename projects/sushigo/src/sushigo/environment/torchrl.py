"""TorchRL integration for the Sushi Go PettingZoo environment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.pettingzoo import PettingZooWrapper

from .sushi_go import OBS_COMPONENTS, SushiGoParallelEnv

GROUP = "players"
OBS_KEYS = tuple((GROUP, "observation", key) for key in OBS_COMPONENTS)
OBS_KEY = (GROUP, "flat_observation")
PLAYER_MASK_KEY = (GROUP, "observation", "player_mask")
HAND_HISTORY_MASK_KEY = (GROUP, "observation", "hand_history_mask")
OPPONENT_TABLEAUS_MASK_KEY = (
    GROUP,
    "observation",
    "opponent_tableaus_mask",
)
MASK_KEY = (GROUP, "action_mask")
ACTION_KEY = (GROUP, "action")
ACTION_VALUE_KEY = (GROUP, "action_value")
CHOSEN_VALUE_KEY = (GROUP, "chosen_action_value")
ENCODER_KEY = (GROUP, "encoder_embedding")


class StructuredObservationFlattener(nn.Module):
    """Flatten model observation components while preserving batch/player axes."""

    def forward(self, *components: torch.Tensor) -> torch.Tensor:
        flat = [
            component.flatten(start_dim=-component_ndims)
            for component, component_ndims in zip(
                components, (1, 2, 1, 2, 1, 1), strict=True
            )
        ]
        return torch.cat(flat, dim=-1)


def make_observation_flattener() -> TensorDictModule:
    return TensorDictModule(
        StructuredObservationFlattener(),
        in_keys=list(OBS_KEYS),
        out_keys=[OBS_KEY],
    )


def flat_observation_dim(environment) -> int:
    return sum(
        int(torch.tensor(environment.observation_spec[key].shape[1:]).prod())
        for key in OBS_KEYS
    )


def resolve_player_counts(
    environment_config: Mapping[str, Any],
) -> tuple[int | None, int | None, int | None, int]:
    fixed = environment_config.get("n_players")
    minimum = environment_config.get("min_n_players")
    maximum = environment_config.get("max_n_players")
    if fixed is not None:
        if minimum is not None or maximum is not None:
            raise ValueError(
                "Use n_players or min_n_players/max_n_players, not both."
            )
        fixed = int(fixed)
        return fixed, None, None, fixed
    if minimum is None or maximum is None:
        raise ValueError(
            "Variable-player environments require min_n_players and max_n_players."
        )
    minimum, maximum = int(minimum), int(maximum)
    if not 2 <= minimum <= maximum <= 4:
        raise ValueError("Player counts must satisfy 2 <= min <= max <= 4.")
    return None, minimum, maximum, maximum


def make_env(
    config: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
    reward_scale: float | None = None,
) -> TransformedEnv:
    """Construct the fixed-shape TorchRL environment described by config."""

    environment_config = config["environment"]
    fixed, minimum, maximum, dense_players = resolve_player_counts(
        environment_config
    )
    base = SushiGoParallelEnv(
        n_players=fixed,
        min_n_players=minimum,
        max_n_players=maximum,
        history_len=environment_config.get("history_len"),
        include_opponent_tableaus=bool(
            environment_config.get("include_opponent_tableaus", True)
        ),
        reward_scale=float(
            environment_config.get("reward_scale", 0.1)
            if reward_scale is None
            else reward_scale
        ),
    )
    wrapped = PettingZooWrapper(
        base,
        use_mask=True,
        categorical_actions=True,
        group_map={
            GROUP: [f"player_{index}" for index in range(dense_players)]
        },
        device=device,
    )
    return TransformedEnv(
        wrapped,
        RewardSum(
            in_keys=[(GROUP, "reward")],
            out_keys=[(GROUP, "episode_reward")],
        ),
    )
