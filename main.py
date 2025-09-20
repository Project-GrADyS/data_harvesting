import torch
from torchrl.envs import check_env_specs, TransformedEnv, RewardSum

from data_harvesting.actor import create_actor, create_exploratory_actor
from data_harvesting.environment import make_env
from data_harvesting.critic import create_critic
from data_harvesting.collector import create_collector
from data_harvesting.metrics import EnvironmentMetricsCollector, LearningMetricsCollector
from data_harvesting.replay import create_replay_buffer
from data_harvesting.optimization import create_loss, create_optimizers, create_updater
from tqdm import tqdm
from dvclive import Live
import yaml

def main():
    with open("params.yaml", "rb") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def transformed_env(check: bool = False) -> TransformedEnv:
        base_env = make_env(config)
        env = TransformedEnv(
            base_env,
            RewardSum(
                in_keys=base_env.reward_keys,
                reset_keys=["_reset"] * len(base_env.group_map.keys()),
            )
        )
        if check:
            check_env_specs(env)
        return env

    sample_env = transformed_env(True)

    policy = create_actor(sample_env, device, config)

    exploratory_policy, exploration_noise = create_exploratory_actor(policy, device, config)

    critic = create_critic(sample_env, device, config)

    collector = create_collector(exploratory_policy, device, transformed_env, config)

    replay_buffer = create_replay_buffer(config, device)

    loss_module = create_loss(policy, critic, config)
    optimizers = create_optimizers(loss_module, config)
    target_updater = create_updater(loss_module, config)

    total_steps = config["training"]["total_timesteps"]
    frames_per_step = config["collector"]["frames_per_batch"]
    n_optimiser_steps = config["optimization"]["num_optimizer_steps"]
    grad_clip = config["optimization"]["grad_clip"]

    pbar = tqdm(total=total_steps)

    with Live() as live:
        live.log_params(config)

        metrics_logger = EnvironmentMetricsCollector(live)
        learning_logger = LearningMetricsCollector(live)

        # Training/collection iterations
        for iteration, batch in enumerate(collector):
            current_frames = batch.numel()
            replay_buffer.extend(batch)

            for _ in range(n_optimiser_steps):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                for loss_name in ["loss_actor", "loss_value"]:
                    loss = loss_vals[loss_name]
                    optimiser: torch.optim.Optimizer = optimizers[loss_name]

                    loss.backward()

                    # Optional
                    if grad_clip > 0:
                        params = optimiser.param_groups[0]["params"]
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)

                    optimiser.step()
                    optimiser.zero_grad()

                    learning_logger.report_loss(loss_name, loss.item())

                # Soft-update the target network
                target_updater.step()

            # Exploration sigma anneal update
            exploration_noise.step(current_frames)

            # Logging
            metrics_logger.log_metrics(batch)
            learning_logger.log_metrics()

            pbar.update(current_frames)
            live.next_step()

if __name__ == "__main__":
    main()