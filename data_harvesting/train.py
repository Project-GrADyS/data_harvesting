from copy import deepcopy

import mlflow
import torch
from mlflow import pytorch as mlflow_pytorch
from torchrl.envs import check_env_specs, TransformedEnv, RewardSum
from torchrl.envs.utils import ExplorationType, set_exploration_type

from data_harvesting.environment import make_env, make_metrics_spec
from data_harvesting.collector import create_collector
from data_harvesting.metrics import EnvironmentMetricsCollector, LearningMetricsCollector
from data_harvesting.algorithm import MADDPGAlgorithm, MAPPOAlgorithm
from tqdm import tqdm

torch.set_float32_matmul_precision('high')


def log_model(algorithm: MADDPGAlgorithm | MAPPOAlgorithm, name: str = "policy_model"):
    policy_copy = deepcopy(algorithm.policy)
    try:
        policy_cpu = policy_copy.to("cpu")
    except Exception:
        # If .to is unsupported for any wrapped module, fall back to original
        policy_cpu = policy_copy
    mlflow_pytorch.log_model(policy_cpu, name=name)


def _maybe_log_checkpoint(
    algorithm: MADDPGAlgorithm | MAPPOAlgorithm,
    config: dict,
    *,
    experience_steps: int,
    last_checkpoint_step: int,
) -> int:
    """
    Checks if it is time to log a checkpoint and does so if it is.

    :param algorithm: Instance of the algorithm to log
    :param config: Configuration dict
    :param experience_steps: Current number of experience steps collected so far
    :param last_checkpoint_step: The experience step count at which the last checkpoint was logged
    :return: The experience step count at which the most recent checkpoint was logged (either the same as
    last_checkpoint_step or updated to experience_steps if a new checkpoint was logged)
    """
    checkpoint_config = config["checkpoint"]
    if not bool(checkpoint_config.get("enabled", True)):
        return last_checkpoint_step

    checkpoint_every_n_steps = int(checkpoint_config["checkpoint_every_n_steps"])
    if checkpoint_every_n_steps <= 0:
        return last_checkpoint_step

    if experience_steps - last_checkpoint_step >= checkpoint_every_n_steps:
        log_model(algorithm, name=f"policy_checkpoint_step_{experience_steps}")
        return experience_steps

    return last_checkpoint_step


def _should_save_final_model(config: dict) -> bool:
    return bool(config["checkpoint"]["save_final_model"])


def _log_prefixed_metrics(logger: EnvironmentMetricsCollector, *, prefix: str, step: int) -> None:
    metrics = logger._build_log_metrics()
    if metrics:
        mlflow.log_metrics({f"{prefix}/{key}": value for key, value in metrics.items()}, step=step)


def _module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _make_cpu_eval_policy(policy: torch.nn.Module) -> torch.nn.Module:
    if _module_device(policy).type == "cpu":
        return policy

    policy_copy = deepcopy(policy)
    try:
        return policy_copy.to("cpu")
    except Exception:
        return policy_copy


def _run_periodic_evaluation(
    algorithm: MADDPGAlgorithm | MAPPOAlgorithm,
    config: dict,
    *,
    experience_steps: int,
    device: torch.device,
    metrics_spec,
    num_runs: int,
    seed: int | None = None,
) -> None:
    if num_runs <= 0:
        return

    eval_config = deepcopy(config)
    eval_config.setdefault("environment", {})["render_mode"] = None
    eval_env = make_env(eval_config)

    eval_device = torch.device("cpu")
    eval_logger = EnvironmentMetricsCollector(eval_device, metrics_spec)
    policy_was_training = algorithm.policy.training
    eval_policy = _make_cpu_eval_policy(algorithm.policy)

    try:
        algorithm.policy.eval()
        if eval_policy is not algorithm.policy:
            eval_policy.eval()
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            for run_index in range(num_runs):
                if seed is not None:
                    eval_env.set_seed(seed + run_index)

                rollout = eval_env.rollout(
                    max_steps=eval_config["environment"]["max_episode_length"],
                    policy=eval_policy,
                )
                eval_logger.report_metrics(rollout.reshape(-1))

        _log_prefixed_metrics(eval_logger, prefix="eval", step=experience_steps)
    finally:
        algorithm.policy.train(policy_was_training)
        if hasattr(eval_env, "close"):
            eval_env.close()


def _maybe_run_periodic_evaluation(
    algorithm: MADDPGAlgorithm | MAPPOAlgorithm,
    config: dict,
    *,
    experience_steps: int,
    last_eval_step: int,
    device: torch.device,
    metrics_spec,
) -> int:
    evaluation_config = config.get("evaluation", {})
    if not bool(evaluation_config.get("enabled", False)):
        return last_eval_step

    eval_every_n_steps = int(evaluation_config.get("eval_every_n_steps", 0))
    if eval_every_n_steps <= 0:
        return last_eval_step

    if experience_steps - last_eval_step >= eval_every_n_steps:
        _run_periodic_evaluation(
            algorithm,
            config,
            experience_steps=experience_steps,
            device=device,
            metrics_spec=metrics_spec,
            num_runs=int(evaluation_config.get("num_runs", 1)),
            seed=evaluation_config.get("seed"),
        )
        return experience_steps

    return last_eval_step


def train(config: dict, run_name: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_spec = make_metrics_spec()

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

    pbar = tqdm(total=total_steps)

    collection_device = config["collector"]["device"]
    with (
        mlflow.start_run(run_name=run_name), 
        create_collector(algorithm.exploratory_policy, collection_device, transformed_env, config) as collector
    ):
        try:
            mlflow.log_params(config)

            metrics_logger = EnvironmentMetricsCollector(device, metrics_spec)
            learning_logger = LearningMetricsCollector(device)

            experience_steps = 0
            last_metric_log = 0
            last_checkpoint_step = 0
            last_eval_step = 0

            # Training/collection iterations
            for iteration, batch in enumerate(collector):
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

                pbar.update(current_frames)
                experience_steps += current_frames
                last_checkpoint_step = _maybe_log_checkpoint(
                    algorithm,
                    config,
                    experience_steps=experience_steps,
                    last_checkpoint_step=last_checkpoint_step,
                )
                last_eval_step = _maybe_run_periodic_evaluation(
                    algorithm,
                    config,
                    experience_steps=experience_steps,
                    last_eval_step=last_eval_step,
                    device=device,
                    metrics_spec=metrics_spec,
                )
            
            # Logging metrics at the end of training
            learning_logger.log_metrics(experience_steps)
            metrics_logger.log_metrics(experience_steps)
        finally:
            if _should_save_final_model(config):
                log_model(algorithm)

    # Returning the final average reward as a simple measure of performance
    # Useful for hyperparameter tuning
    if "avg_reward" in metrics_logger.scalar_totals:
        return metrics_logger.metric_value("avg_reward")
    else:
        raise Exception("A avg_reward metric is required to report training results")
