from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from time import perf_counter
from typing import Any

import mlflow
import torch
from mlflow import pytorch as mlflow_pytorch
from rl_core import CollectionMode, CollectorConfig, make_collector
from torchrl.envs import check_env_specs
from tqdm import tqdm

from kaz_training.algorithm import MADDPG
from kaz_training.config import flatten_config, resolve_device, validate_config
from kaz_training.environment import make_env
from kaz_training.evaluation import evaluate


def _log_policy(policy: torch.nn.Module, artifact_path: str) -> None:
    policy_copy = deepcopy(policy).to("cpu")
    mlflow_pytorch.log_model(policy_copy, name=artifact_path)


def _completed_episode_metrics(batch) -> dict[str, float]:
    done = batch.get(("next", "done")).squeeze(-1).to(torch.bool)
    if not bool(done.any()):
        return {}
    returns = batch.get(("next", "episode_team_reward"))[done]
    lengths = batch.get(("next", "step_count"))[done]
    return {
        "train/team_kills": float(returns.float().mean()),
        "train/episode_length": float(lengths.float().mean()),
    }


def _collector_config(config: Mapping[str, Any]) -> CollectorConfig:
    collector = config["collector"]
    return CollectorConfig(
        mode=CollectionMode.SYNC,
        frames_per_batch=int(collector["frames_per_batch"]),
        total_frames=int(config["training"]["total_timesteps"]),
        num_workers=int(collector["num_collectors"]),
        device=str(collector["device"]),
        env_device="cpu",
        policy_device=str(collector["device"]),
    )


def train(config: dict[str, Any], run_name: str | None = None) -> float:
    validate_config(config)
    torch.manual_seed(int(config["training"]["seed"]))
    device = resolve_device(config)

    sample_env = make_env(config)
    try:
        check_env_specs(sample_env)
        algorithm = MADDPG(sample_env, config, device)
    except Exception:
        sample_env.close()
        raise

    total_steps = int(config["training"]["total_timesteps"])
    metric_interval = int(config["metrics"]["log_every_n_steps"])
    evaluation_config = config["evaluation"]
    eval_interval = int(evaluation_config["eval_every_n_steps"])
    checkpoint_config = config["checkpoint"]
    checkpoint_interval = int(checkpoint_config["every_n_steps"])
    experience_steps = 0
    next_metric = metric_interval
    next_eval = eval_interval
    next_checkpoint = checkpoint_interval
    latest_losses: dict[str, float] = {}
    progress = tqdm(total=total_steps, desc="KAZ training", unit="frame")

    try:
        with (
            mlflow.start_run(run_name=run_name),
            make_collector(
                config=_collector_config(config),
                env_factory=lambda: make_env(config),
                policy=algorithm.exploratory_policy,
            ) as collector,
        ):
            mlflow.log_params(flatten_config(config))
            for batch in collector:
                current_frames = batch.numel()
                started = perf_counter()
                latest_losses = algorithm.learn(batch.reshape(-1))
                elapsed = max(perf_counter() - started, 1e-9)
                collector.update_policy_weights_()

                experience_steps += current_frames
                progress.update(current_frames)
                episode_metrics = _completed_episode_metrics(batch)

                if metric_interval > 0 and experience_steps >= next_metric:
                    metrics = {
                        **latest_losses,
                        **episode_metrics,
                        "train/frames_per_second": current_frames / elapsed,
                        "train/replay_size": float(len(algorithm.replay_buffer)),
                        "train/epsilon": algorithm.epsilon,
                    }
                    mlflow.log_metrics(metrics, step=experience_steps)
                    while next_metric <= experience_steps:
                        next_metric += metric_interval

                if (
                    bool(evaluation_config["enabled"])
                    and eval_interval > 0
                    and experience_steps >= next_eval
                ):
                    evaluation = evaluate(
                        algorithm.policy,
                        config,
                        num_episodes=int(evaluation_config["num_episodes"]),
                        seed=evaluation_config.get("seed"),
                    )
                    mlflow.log_metrics(
                        {f"eval/{key}": value for key, value in evaluation.items()},
                        step=experience_steps,
                    )
                    while next_eval <= experience_steps:
                        next_eval += eval_interval

                if (
                    bool(checkpoint_config["enabled"])
                    and checkpoint_interval > 0
                    and experience_steps >= next_checkpoint
                ):
                    _log_policy(
                        algorithm.policy,
                        f"policy_checkpoint_step_{experience_steps}",
                    )
                    while next_checkpoint <= experience_steps:
                        next_checkpoint += checkpoint_interval

            final_evaluation = evaluate(
                algorithm.policy,
                config,
                num_episodes=int(evaluation_config["num_episodes"]),
                seed=evaluation_config.get("seed"),
            )
            mlflow.log_metrics(
                {f"eval/final_{key}": value for key, value in final_evaluation.items()},
                step=experience_steps,
            )
            if bool(checkpoint_config["save_final_model"]):
                _log_policy(algorithm.policy, "policy_model")
            return final_evaluation["team_kills_mean"]
    finally:
        progress.close()
        sample_env.close()
