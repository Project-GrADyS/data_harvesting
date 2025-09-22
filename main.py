import torch
from torchrl.envs import check_env_specs, TransformedEnv, RewardSum

from data_harvesting.actor import create_actor, create_exploratory_actor
from data_harvesting.environment import make_env
from data_harvesting.critic import create_critic
from data_harvesting.collector import create_collector
from data_harvesting.metrics import EnvironmentMetricsCollector, LearningMetricsCollector
from data_harvesting.algorithm import MADDPGAlgorithm, MAPPOAlgorithm
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
    algo_name = config["training"]["algorithm"].lower()
    if algo_name == "mappo":
        algorithm = MAPPOAlgorithm(sample_env, device, config)
    else:
        algorithm = MADDPGAlgorithm(sample_env, device, config)
    collector = create_collector(algorithm.exploratory_policy, device, transformed_env, config)
    total_steps = config["training"]["total_timesteps"]

    pbar = tqdm(total=total_steps)

    with Live() as live:
        live.log_params(config)

        metrics_logger = EnvironmentMetricsCollector(live)
        learning_logger = LearningMetricsCollector(live)

        # Training/collection iterations
        for iteration, batch in enumerate(collector):
            current_frames = batch.numel()
            
            # Learning step
            losses = algorithm.learn(batch)

            for loss_name, loss_value in losses.items():
                live.log_metric(loss_name, loss_value)

            # Logging
            metrics_logger.log_metrics(batch)
            learning_logger.log_metrics()

            pbar.update(current_frames)
            live.next_step()

if __name__ == "__main__":
    main()