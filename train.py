import mlflow
import torch
from torchrl.envs import check_env_specs, TransformedEnv, RewardSum

from data_harvesting.environment import make_env
from data_harvesting.collector import create_collector
from data_harvesting.metrics import EnvironmentMetricsCollector, LearningMetricsCollector
from data_harvesting.algorithm import MADDPGAlgorithm, MAPPOAlgorithm
from tqdm import tqdm

mlflow.set_tracking_uri("file:./mlruns")

def train(config: dict):
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
    log_every_n_steps = config["metrics"]["log_every_n_steps"]

    pbar = tqdm(total=total_steps)

    with mlflow.start_run():
        mlflow.log_params(config)

        metrics_logger = EnvironmentMetricsCollector()
        learning_logger = LearningMetricsCollector()

        # Training/collection iterations
        for iteration, batch in enumerate(collector):
            current_frames = batch.numel()
            
            # Learning step
            losses = algorithm.learn(batch)
            for loss_name, loss_value in losses.items():
                learning_logger.report_loss(loss_name, loss_value)
            metrics_logger.report_metrics(batch)
            
            # Logging
            if iteration % log_every_n_steps == 0:
                learning_logger.log_metrics(iteration)
                metrics_logger.log_metrics(iteration)

            pbar.update(current_frames)