import mlflow
import torch
from pathlib import Path
from torchrl.envs import check_env_specs, TransformedEnv, RewardSum

from data_harvesting.environment import make_env
from data_harvesting.collector import create_collector
from data_harvesting.metrics import EnvironmentMetricsCollector, LearningMetricsCollector
from data_harvesting.algorithm import MADDPGAlgorithm, MAPPOAlgorithm
from data_harvesting.checkpoint import save_checkpoint, load_checkpoint
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

def save_model(algorithm: MADDPGAlgorithm | MAPPOAlgorithm):
    # Move to CPU for portability when loading in environments without CUDA
    try:
        policy_cpu = algorithm.policy.to("cpu")
    except Exception:
        # If .to is unsupported for any wrapped module, fall back to original
        policy_cpu = algorithm.policy
    mlflow.pytorch.log_model(policy_cpu, name="policy_model")

def train(config: dict, run_name: str | None = None):
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

    total_steps = config["training"]["total_timesteps"]
    log_every_n_steps = config["metrics"]["log_every_n_steps"]
    
    # Checkpointing configuration
    checkpoint_enabled = config.get("checkpointing", {}).get("enabled", False)
    checkpoint_interval = config.get("checkpointing", {}).get("checkpoint_interval", 50000)
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", "checkpoints")
    resume_from = config.get("checkpointing", {}).get("resume_from", None)

    pbar = tqdm(total=total_steps)

    collection_device = config["collector"]["device"]
    with (
        mlflow.start_run(run_name=run_name), 
        create_collector(algorithm.exploratory_policy, collection_device, transformed_env, config) as collector
    ):
        try:
            mlflow.log_params(config)

            metrics_logger = EnvironmentMetricsCollector(device)
            learning_logger = LearningMetricsCollector(device)

            experience_steps = 0
            last_metric_log = 0
            iteration = 0
            last_checkpoint_steps = 0
            
            # Load checkpoint if resuming
            if resume_from and Path(resume_from).exists():
                checkpoint_data = load_checkpoint(resume_from, algorithm, metrics_logger, learning_logger)
                experience_steps = checkpoint_data["experience_steps"]
                iteration = checkpoint_data["iteration"]
                last_checkpoint_steps = experience_steps
                pbar.update(experience_steps)
                print(f"Resumed from checkpoint: {resume_from} at step {experience_steps}")

            # Training/collection iterations
            for batch in collector:
                current_frames = batch.numel()
                # The batch shape is (num_collectors, frames_per_batch, ...), flatten it to a single batch dimension
                batch = batch.reshape(-1)
                
                # Learning step
                losses = algorithm.learn(batch)
                for loss_name, loss_value in losses.items():
                    learning_logger.report_loss(loss_name, loss_value)
                metrics_logger.report_metrics(batch)

                # Sync updated policy weights to collector workers.
                # On CUDA this is a no-op (workers share GPU memory via CUDA IPC),
                # but on CPU workers hold independent copies that must be
                # explicitly refreshed after each training step.
                collector.update_policy_weights_()
                
                # Logging
                if experience_steps - last_metric_log > log_every_n_steps:
                    learning_logger.log_metrics(experience_steps)
                    metrics_logger.log_metrics(experience_steps)
                    last_metric_log = experience_steps
                
                # Checkpointing
                if checkpoint_enabled and experience_steps - last_checkpoint_steps >= checkpoint_interval:
                    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_step_{experience_steps}.pt"
                    save_checkpoint(
                        checkpoint_path,
                        algorithm,
                        experience_steps,
                        iteration,
                        metrics_logger,
                        learning_logger
                    )
                    last_checkpoint_steps = experience_steps
                    print(f"Checkpoint saved at step {experience_steps}")

                pbar.update(current_frames)
                experience_steps += current_frames
                iteration += 1
            
            # Logging metrics at the end of training
            learning_logger.log_metrics(experience_steps)
            metrics_logger.log_metrics(experience_steps)
        finally:
            if config["metrics"]["save_model"]:
                save_model(algorithm)

    # Returning the final average reward as a simple measure of performance
    # Useful for hyperparameter tuning
    avg_reward = (metrics_logger.sum_avg_reward / metrics_logger.trajectories).item()
    return avg_reward