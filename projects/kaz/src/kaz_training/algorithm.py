from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torchrl.data import LazyTensorStorage, RandomSampler, ReplayBuffer
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators

from kaz_training.models import create_actor, create_critic, create_exploratory_actor


class MADDPG:
    def __init__(self, env, config: Mapping[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        self.policy = create_actor(env, config, device)
        self.exploratory_policy, self.explorer = create_exploratory_actor(
            self.policy, env, config, device
        )
        self.critic = create_critic(env, config, device)
        self.loss_module = DDPGLoss(
            actor_network=self.policy,
            value_network=self.critic,
            delay_actor=True,
            delay_value=True,
            loss_function="l2",
        )
        self.loss_module.set_keys(
            state_action_value="state_action_value",
            reward="team_reward",
            done="done",
            terminated="terminated",
        )
        self.loss_module.make_value_estimator(
            ValueEstimators.TD0,
            gamma=float(config["optimization"]["gamma"]),
            device=device,
        )
        lr = float(config["optimization"]["lr"])
        self.actor_optimizer = torch.optim.Adam(
            self.loss_module.actor_network_params.flatten_keys().values(), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.loss_module.value_network_params.flatten_keys().values(), lr=lr
        )
        self.target_updater = SoftUpdate(
            self.loss_module, tau=float(config["optimization"]["tau"])
        )
        replay_config = config["replay_buffer"]
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                int(replay_config["buffer_size"]), device=str(replay_config["device"])
            ),
            sampler=RandomSampler(),
            batch_size=int(config["training"]["batch_size"]),
            prefetch=int(replay_config["prefetch"]),
        )
        self.replay_buffer.append_transform(lambda td: td.to(device))
        self.warmup_steps = max(
            int(config["training"]["warmup_steps"]),
            int(config["training"]["batch_size"]),
        )
        self.updates_per_batch = int(config["optimization"]["updates_per_batch"])
        self.grad_clip = float(config["optimization"]["grad_clip"])

    @property
    def epsilon(self) -> float:
        return float(self.explorer.eps.item())

    def learn(self, batch) -> dict[str, float]:
        current_frames = batch.numel()
        self.replay_buffer.extend(batch)
        self.explorer.step(current_frames)
        if len(self.replay_buffer) < self.warmup_steps or self.updates_per_batch <= 0:
            return {}

        totals = {"loss_actor": 0.0, "loss_value": 0.0}
        for _ in range(self.updates_per_batch):
            sample = self.replay_buffer.sample()
            losses = self.loss_module(sample)

            actor_loss = losses["loss_actor"]
            actor_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.actor_optimizer.param_groups[0]["params"], self.grad_clip
                )
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad(set_to_none=True)

            value_loss = losses["loss_value"]
            value_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.critic_optimizer.param_groups[0]["params"], self.grad_clip
                )
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)

            self.target_updater.step()
            totals["loss_actor"] += float(actor_loss.detach())
            totals["loss_value"] += float(value_loss.detach())

        return {key: value / self.updates_per_batch for key, value in totals.items()}
