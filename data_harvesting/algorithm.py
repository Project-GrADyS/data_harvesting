
import torch
from typing import Iterator
from torchrl.envs import EnvBase


from data_harvesting.actor import create_actor, create_exploratory_actor, create_ppo_actor
from data_harvesting.critic import create_critic, create_value_net
from data_harvesting.replay import create_replay_buffer
from data_harvesting.optimization import (
    create_loss,
    create_optimizers,
    create_updater,
    create_ppo_loss,
    create_ppo_optimizers,
)

class MADDPGAlgorithm:
    def __init__(self, env: EnvBase, device: torch.device, config: dict):
        self.config = config
        self.device = device
        self.env = env

        self.policy = create_actor(env, device, config)
        self.exploratory_policy, self.exploration_noise = create_exploratory_actor(self.policy, device, config)
        self.critic = create_critic(env, device, config)
        self.replay_buffer = create_replay_buffer(config, device)
        self.loss_module = create_loss(self.policy, self.critic, config)
        self.optimizers = create_optimizers(self.loss_module, config)
        self.target_updater = create_updater(self.loss_module, config)

        self.n_optimiser_steps = config["optimization"]["num_optimizer_steps"]
        self.grad_clip = config["optimization"]["grad_clip"]

    def learn(self, batch):
        current_frames = batch.numel()
        self.replay_buffer.extend(batch)

        loss_sums = {"loss_actor": 0.0, "loss_value": 0.0}
        for _ in range(self.n_optimiser_steps):
            subdata = self.replay_buffer.sample()
            loss_vals = self.loss_module(subdata)
            for loss_name in ["loss_actor", "loss_value"]:
                loss = loss_vals[loss_name]
                optimiser: torch.optim.Optimizer = self.optimizers[loss_name]

                loss.backward()

                if self.grad_clip > 0:
                    params = optimiser.param_groups[0]["params"]
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

                optimiser.step()
                optimiser.zero_grad(set_to_none=True)

                loss_sums[loss_name] += loss.detach().item()

            self.target_updater.step()

        self.exploration_noise.step(current_frames)

        avg_losses = {name: loss_sums[name] / self.n_optimiser_steps for name in loss_sums}
        return avg_losses


class MAPPOAlgorithm:
    def __init__(self, env: EnvBase, device: torch.device, config: dict):
        self.config = config
        self.device = device
        self.env = env

        self.policy = create_ppo_actor(env, device, config)
        # For collection, exploration is inherent in stochastic policy; no extra noise needed
        self.exploratory_policy = self.policy
        self.critic = create_value_net(env, device, config)
        self.loss_module = create_ppo_loss(self.policy, self.critic, config)
        self.optimizers = create_ppo_optimizers(self.loss_module, config)

        ppo_cfg = config["ppo"]
        self.num_epochs = ppo_cfg["num_epochs"]
        self.minibatch_size = ppo_cfg["minibatch_size"]
        self.max_grad_norm = config["optimization"]["grad_clip"]

    def _iterate_minibatches(self, batch) -> Iterator:
        # Flatten leading dims (time, env) into single dimension per agent
        td = batch
        # TorchRL tensordict supports .reshape(-1) via td.view; using td.reshape is not available
        flattened = td.reshape(-1)
        n_items = flattened.batch_size[0]
        indices = torch.randperm(n_items, device=self.device)
        # indices refers to number of sub-tensordicts; we can use split
        for start in range(0, n_items, self.minibatch_size):
            yield flattened[indices[start:start + self.minibatch_size]]

    def learn(self, batch):
        # Compute GAE once on the full batch
        with torch.no_grad():
            self.loss_module.value_estimator(batch)

        loss_sums = {"loss_policy": 0.0, "loss_value": 0.0}
        n_steps = 0

        for _ in range(self.num_epochs):
            for mb in self._iterate_minibatches(batch):
                # policy step
                loss_td = self.loss_module(mb)
                policy_loss = loss_td["loss_objective"]
                policy_loss.backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    params = self.optimizers["loss_policy"].param_groups[0]["params"]
                    torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizers["loss_policy"].step()
                self.optimizers["loss_policy"].zero_grad(set_to_none=True)

                # value step (recompute forward to keep graphs separate)
                loss_td = self.loss_module(mb)
                value_loss = loss_td["loss_critic"]
                value_loss.backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    params_v = self.optimizers["loss_value"].param_groups[0]["params"]
                    torch.nn.utils.clip_grad_norm_(params_v, self.max_grad_norm)
                self.optimizers["loss_value"].step()
                self.optimizers["loss_value"].zero_grad(set_to_none=True)

                loss_sums["loss_policy"] += policy_loss.detach().item()
                loss_sums["loss_value"] += value_loss.detach().item()
                n_steps += 1

        if n_steps == 0:
            return {"loss_policy": 0.0, "loss_value": 0.0}
        return {k: v / n_steps for k, v in loss_sums.items()}



