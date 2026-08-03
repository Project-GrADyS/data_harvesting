"""Independent shared-parameter DQN implementation for Sushi Go."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.data import LazyTensorStorage, RandomSampler, ReplayBuffer
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, SoftUpdate, ValueEstimators

from .environment.torchrl import (
    ACTION_KEY,
    ACTION_VALUE_KEY,
    CHOSEN_VALUE_KEY,
    GROUP,
    MASK_KEY,
    PLAYER_MASK_KEY,
)
from .policy import build_q_policy


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.squeeze(-1).to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


class DQNAlgorithm:
    """Own policy, exploration, replay, loss, optimizer, and target updates."""

    def __init__(
        self,
        environment,
        config: Mapping[str, Any],
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.policy = build_q_policy(environment, config, device=device)
        self.policy(environment.reset())

        exploration_config = config["exploration"]
        self.exploration = EGreedyModule(
            spec=environment.action_spec,
            eps_init=float(exploration_config["eps_init"]),
            eps_end=float(exploration_config["eps_end"]),
            annealing_num_steps=int(exploration_config["annealing_steps"]),
            action_key=ACTION_KEY,
            action_mask_key=MASK_KEY,
        )
        self.exploratory_policy = TensorDictSequential(
            self.policy, self.exploration
        )

        optimization = config["optimization"]
        self.loss_module = DQNLoss(
            self.policy,
            action_space="categorical",
            delay_value=True,
            reduction="none",
        )
        self.loss_module.set_keys(
            action_value=ACTION_VALUE_KEY,
            action=ACTION_KEY,
            value=CHOSEN_VALUE_KEY,
            reward=(GROUP, "reward"),
            done=(GROUP, "done"),
            terminated=(GROUP, "terminated"),
        )
        self.loss_module.make_value_estimator(
            ValueEstimators.TD0, gamma=float(optimization["gamma"])
        )
        self.target_updater = SoftUpdate(
            self.loss_module, eps=float(optimization["target_eps"])
        )
        self.optimizer = torch.optim.Adam(
            self.loss_module.parameters(), lr=float(optimization["lr"])
        )
        self.grad_clip = float(optimization["max_grad_norm"])
        self.updates_per_batch = int(optimization["updates_per_batch"])

        replay_config = config["replay_buffer"]
        self.replay = ReplayBuffer(
            storage=LazyTensorStorage(
                int(replay_config["capacity"]),
                device=str(replay_config["device"]),
            ),
            sampler=RandomSampler(),
            batch_size=int(replay_config["batch_size"]),
            prefetch=int(replay_config.get("prefetch", 0)),
        )

    @property
    def epsilon(self) -> float:
        value = self.exploration.eps
        return float(value.item() if hasattr(value, "item") else value)

    def learn(self, batch: TensorDictBase) -> dict[str, torch.Tensor | float]:
        flattened = batch.reshape(-1)
        self.replay.extend(flattened.cpu())
        losses = torch.zeros((), device=self.device)
        for _ in range(self.updates_per_batch):
            sample = self.replay.sample().to(self.device)
            loss_values = self.loss_module(sample)
            loss = masked_mean(
                loss_values["loss"], sample.get(PLAYER_MASK_KEY)
            )
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(), self.grad_clip
                )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.target_updater.step()
            losses += loss.detach()

        frames = batch.numel()
        self.exploration.step(frames)
        return {
            "loss": losses / self.updates_per_batch,
            "epsilon": self.epsilon,
        }
