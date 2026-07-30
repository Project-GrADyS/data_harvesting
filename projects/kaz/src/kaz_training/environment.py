from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium.spaces as spaces
import numpy as np
import torch
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.wrappers import BaseParallelWrapper
from torchrl.data import Composite, Unbounded
from torchrl.envs import RewardSum, StepCounter, TransformedEnv
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import Compose, Transform


class PaddedSequenceWrapper(BaseParallelWrapper):
    """Turn KAZ's experimental Sequence observations into fixed tensors and masks."""

    def __init__(self, env, *, max_entities: int) -> None:
        super().__init__(env)
        self.metadata = env.metadata
        self.max_entities = max_entities
        self._observation_space = spaces.Dict(
            {
                "entities": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(max_entities, 11),
                    dtype=np.float32,
                ),
                "entity_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(max_entities,),
                    dtype=np.bool_,
                ),
                "role": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            }
        )

    def observation_space(self, agent):
        return self._observation_space

    @staticmethod
    def _role(agent: str) -> np.ndarray:
        if agent.startswith("archer_"):
            return np.asarray([1.0, 0.0], dtype=np.float32)
        if agent.startswith("knight_"):
            return np.asarray([0.0, 1.0], dtype=np.float32)
        raise ValueError(f"Unknown KAZ agent name: {agent}")

    def _convert(self, agent: str, observation: np.ndarray) -> dict[str, np.ndarray]:
        length = int(observation.shape[0])
        if length > self.max_entities:
            raise RuntimeError(
                f"KAZ returned {length} entities, exceeding configured maximum {self.max_entities}."
            )
        entities = np.zeros((self.max_entities, 11), dtype=np.float32)
        entities[:length] = observation
        entity_mask = np.zeros(self.max_entities, dtype=np.bool_)
        entity_mask[:length] = True
        return {
            "entities": entities,
            "entity_mask": entity_mask,
            "role": self._role(agent),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return {
            agent: self._convert(agent, observation)
            for agent, observation in observations.items()
        }, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return (
            {
                agent: self._convert(agent, observation)
                for agent, observation in observations.items()
            },
            rewards,
            terminations,
            truncations,
            infos,
        )


class TeamReward(Transform):
    """Add a scalar cooperative reward by summing agent rewards."""

    def __init__(self) -> None:
        super().__init__(
            in_keys=[("agents", "reward")],
            out_keys=["team_reward"],
        )

    def _step(self, tensordict, next_tensordict):
        reward = next_tensordict.get(("agents", "reward"))
        next_tensordict.set("team_reward", reward.sum(dim=-2))
        return next_tensordict

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        output_spec = output_spec.clone()
        reward_spec = output_spec["full_reward_spec"]
        reward_spec["team_reward"] = Unbounded(
            shape=(*reward_spec.shape, 1),
            dtype=torch.float32,
            device=reward_spec.device,
        )
        return output_spec


def max_entities(environment_config: Mapping[str, Any]) -> int:
    return (
        1
        + int(environment_config["num_archers"])
        + int(environment_config["num_knights"])
        + int(environment_config["max_zombies"])
        + int(environment_config["num_knights"])
        + int(environment_config["max_arrows"])
    )


def _make_pettingzoo_env(environment_config: Mapping[str, Any], *, render_mode=None):
    raw = knights_archers_zombies_v10.parallel_env(
        spawn_rate=int(environment_config["spawn_rate"]),
        num_archers=int(environment_config["num_archers"]),
        num_knights=int(environment_config["num_knights"]),
        max_zombies=int(environment_config["max_zombies"]),
        max_arrows=int(environment_config["max_arrows"]),
        killable_knights=bool(environment_config["killable_knights"]),
        killable_archers=bool(environment_config["killable_archers"]),
        pad_observation=bool(environment_config["pad_observation"]),
        line_death=bool(environment_config["line_death"]),
        max_cycles=int(environment_config["max_cycles"]),
        vector_state=True,
        use_typemasks=True,
        sequence_space=True,
        render_mode=render_mode,
    )
    return PaddedSequenceWrapper(raw, max_entities=max_entities(environment_config))


def make_env(config: Mapping[str, Any], *, render_mode=None) -> TransformedEnv:
    environment_config = config["environment"]
    pettingzoo_env = _make_pettingzoo_env(environment_config, render_mode=render_mode)
    group_map = {"agents": list(pettingzoo_env.possible_agents)}
    base = PettingZooWrapper(
        env=pettingzoo_env,
        group_map=group_map,
        use_mask=True,
        categorical_actions=False,
        done_on_any=False,
        seed=int(config["training"]["seed"]),
    )
    return TransformedEnv(
        base,
        Compose(
            TeamReward(),
            RewardSum(
                in_keys=["team_reward"],
                out_keys=["episode_team_reward"],
                reset_keys=["_reset"],
            ),
            StepCounter(max_steps=int(environment_config["max_cycles"])),
        ),
    )
