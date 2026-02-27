import mlflow
import torch
from copy import deepcopy
from mlflow import pytorch as mlflow_pytorch
from torchrl.envs import check_env_specs, TransformedEnv, RewardSum

from data_harvesting.environment import make_env
from data_harvesting.collector import create_collector
from data_harvesting.metrics import EnvironmentMetricsCollector, LearningMetricsCollector
from data_harvesting.algorithm import MADDPGAlgorithm, MAPPOAlgorithm
from data_harvesting.checkpoint import load_checkpoint, save_checkpoint
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

def save_model(algorithm: MADDPGAlgorithm | MAPPOAlgorithm):
    # Log a CPU copy for portability without mutating the live training module.
    try:
        policy_cpu = deepcopy(algorithm.policy).to("cpu")
    except Exception:
        # Fall back to the original object if deepcopy/.to isn't supported.
        policy_cpu = algorithm.policy
    mlflow_pytorch.log_model(policy_cpu, name="policy_model")


def save_checkpoint_policy_model(algorithm: MADDPGAlgorithm | MAPPOAlgorithm):
    try:
        policy_cpu = deepcopy(algorithm.policy).to("cpu")
    except Exception:
        policy_cpu = algorithm.policy
    mlflow_pytorch.log_model(policy_cpu, name="policy_checkpoint")

def train(
    config: dict,
    run_name: str | None = None,
    resume_checkpoint: str | None = None,
    resume_run_id: str | None = None,
):
    if run_name and resume_run_id:
        raise ValueError("run_name (-R) cannot be used together with resume_run_id")
    if resume_run_id and not resume_checkpoint:
        raise ValueError("resume_checkpoint is required when resume_run_id is set")
    if resume_checkpoint and not resume_run_id:
        raise ValueError("resume_run_id is required when resume_checkpoint is set")

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

    pbar = tqdm(total=total_steps)

    collection_device = config["collector"]["device"]
    if resume_run_id:
        run_context = mlflow.start_run(run_id=resume_run_id)
    elif run_name:
        run_context = mlflow.start_run(run_name=run_name)
    else:
        run_context = mlflow.start_run()

    with (
        run_context,
        create_collector(algorithm.exploratory_policy, collection_device, transformed_env, config) as collector
    ):
        try:
            if not resume_run_id:
                mlflow.log_params(config)

            metrics_logger = EnvironmentMetricsCollector(device)
            learning_logger = LearningMetricsCollector(device)

            experience_steps = 0
            last_metric_log = 0
            iteration = 0
            last_checkpoint_steps = 0

            if resume_run_id:
                assert resume_checkpoint is not None
                checkpoint_data = load_checkpoint(
                    resume_run_id,
                    resume_checkpoint,
                    algorithm,
                    metrics_logger,
                    learning_logger,
                )
                experience_steps = checkpoint_data["experience_steps"]
                iteration = checkpoint_data["iteration"]
                last_checkpoint_steps = (experience_steps // checkpoint_interval) * checkpoint_interval
                pbar.update(experience_steps)
                print(f"Resumed training from checkpoint at step {experience_steps}")

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
                    checkpoint_artifact_path = save_checkpoint(
                        algorithm,
                        experience_steps,
                        iteration,
                        metrics_logger,
                        learning_logger,
                    )
                    save_checkpoint_policy_model(algorithm)
                    last_checkpoint_steps = experience_steps
                    print(f"Checkpoint saved at step {experience_steps}: {checkpoint_artifact_path}")

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