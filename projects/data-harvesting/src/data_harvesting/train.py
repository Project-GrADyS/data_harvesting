from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any
from collections.abc import Callable

import mlflow
import torch
from mlflow import pytorch as mlflow_pytorch
from rl_core import (
    EvaluationConfig,
    Evaluator,
    MLflowMetricLogger,
    MetricsCollector,
    ScalarMetricSpec,
    ScalarReducer,
    Scheduler,
)
from torchrl.envs import RewardSum, TransformedEnv, check_env_specs
from tqdm import tqdm

from data_harvesting.algorithm import MADDPGAlgorithm, MAPPOAlgorithm
from data_harvesting.collector import create_collector
from data_harvesting.environment import make_env, make_metrics_spec
from data_harvesting.metrics import (
    extract_selected_terminal_metric_values,
    extract_terminal_metric_values,
)

torch.set_float32_matmul_precision("high")

Algorithm = MADDPGAlgorithm | MAPPOAlgorithm


def log_model(algorithm: Algorithm, name: str = "policy_model") -> None:
    policy_copy = deepcopy(algorithm.policy)
    try:
        policy_copy = policy_copy.to("cpu")
    except Exception:
        pass
    mlflow_pytorch.log_model(policy_copy, name=name)


def _should_save_final_model(config: dict[str, Any]) -> bool:
    return bool(config["checkpoint"]["save_final_model"])


def _module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _make_cpu_eval_policy(policy: torch.nn.Module) -> torch.nn.Module:
    policy_copy = deepcopy(policy)
    if _module_device(policy_copy).type != "cpu":
        try:
            policy_copy = policy_copy.to("cpu")
        except Exception:
            pass
    return policy_copy


class _SeededRolloutEnvironment:
    def __init__(self, environment, seed: int | None) -> None:
        self._environment = environment
        self._seed = seed
        self._episode = 0

    def rollout(self, *args, **kwargs):
        if self._seed is not None:
            self._environment.set_seed(self._seed + self._episode)
        self._episode += 1
        return self._environment.rollout(*args, **kwargs)

    def close(self) -> None:
        self._environment.close()


def _run_periodic_evaluation(
    algorithm: Algorithm,
    config: dict[str, Any],
    *,
    experience_steps: int,
    metrics_spec,
    num_runs: int,
    seed: int | None = None,
) -> dict[str, float]:
    if num_runs <= 0:
        return {}

    eval_config = deepcopy(config)
    eval_config.setdefault("environment", {})["render_mode"] = None
    eval_policy = _make_cpu_eval_policy(algorithm.policy)
    metrics = MetricsCollector(
        specs=metrics_spec,
        loggers=(MLflowMetricLogger(prefix="eval"),),
        device="cpu",
    )

    evaluator = Evaluator(
        config=EvaluationConfig(
            num_episodes=num_runs,
            max_steps=int(eval_config["environment"]["max_episode_length"]),
        ),
        env_factory=lambda: _SeededRolloutEnvironment(make_env(eval_config), seed),
        policy=eval_policy,
        metrics=metrics,
        metric_extractor=lambda terminal: extract_selected_terminal_metric_values(
            terminal, metrics_spec
        ),
    )
    return evaluator.run(experience_steps)


def _make_algorithm(config: dict[str, Any], env: TransformedEnv, device: torch.device) -> Algorithm:
    if config["training"]["algorithm"].lower() == "mappo":
        return MAPPOAlgorithm(env, device, config)
    return MADDPGAlgorithm(env, device, config)


def _configure_scheduler(
    config: dict[str, Any],
    *,
    metrics_callback: Callable[[int], object],
    checkpoint_callback: Callable[[int], object],
    evaluation_callback: Callable[[int], object],
) -> Scheduler:
    scheduler = Scheduler()
    metric_interval = int(config["metrics"]["log_every_n_steps"])
    if metric_interval > 0:
        scheduler.register("metrics", every=metric_interval, callback=metrics_callback)

    checkpoint_config = config["checkpoint"]
    checkpoint_interval = int(checkpoint_config["checkpoint_every_n_steps"])
    if bool(checkpoint_config.get("enabled", True)) and checkpoint_interval > 0:
        scheduler.register(
            "checkpoint", every=checkpoint_interval, callback=checkpoint_callback
        )

    evaluation_config = config.get("evaluation", {})
    evaluation_interval = int(evaluation_config.get("eval_every_n_steps", 0))
    if bool(evaluation_config.get("enabled", False)) and evaluation_interval > 0:
        scheduler.register(
            "evaluation", every=evaluation_interval, callback=evaluation_callback
        )
    return scheduler


def train(config: dict[str, Any], run_name: str | None = None, profiler=None) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_spec = make_metrics_spec()

    def transformed_env(check: bool = False) -> TransformedEnv:
        base_env = make_env(config)
        environment = TransformedEnv(
            base_env,
            RewardSum(
                in_keys=base_env.reward_keys,
                reset_keys=["_reset"] * len(base_env.group_map),
            ),
        )
        if check:
            check_env_specs(environment)
        return environment

    sample_env = transformed_env(check=True)
    algorithm = _make_algorithm(config, sample_env, device)
    total_steps = int(config["training"]["total_timesteps"])
    progress = tqdm(total=total_steps)

    environment_metrics = MetricsCollector(
        specs=metrics_spec,
        loggers=(MLflowMetricLogger(),),
        device=device,
    )
    final_performance_metrics = MetricsCollector(
        specs=(ScalarMetricSpec(key="avg_reward", reducer=ScalarReducer.MEAN),),
        device=device,
    )
    learning_metrics = MetricsCollector(
        specs=(
            ScalarMetricSpec(key="loss_actor", reducer=ScalarReducer.MEAN, output_name="loss_loss_actor"),
            ScalarMetricSpec(key="loss_policy", reducer=ScalarReducer.MEAN, output_name="loss_loss_policy"),
            ScalarMetricSpec(key="loss_value", reducer=ScalarReducer.MEAN, output_name="loss_loss_value"),
            ScalarMetricSpec(key="sps", reducer=ScalarReducer.MEAN),
        ),
        loggers=(MLflowMetricLogger(),),
        device=device,
    )

    evaluation_config = config.get("evaluation", {})
    scheduler = _configure_scheduler(
        config,
        metrics_callback=lambda step: (
                learning_metrics.flush(step=step),
                environment_metrics.flush(step=step),
        ),
        checkpoint_callback=lambda step: log_model(
            algorithm, name=f"policy_checkpoint_step_{step}"
        ),
        evaluation_callback=lambda step: _run_periodic_evaluation(
            algorithm,
            config,
            experience_steps=step,
            metrics_spec=metrics_spec,
            num_runs=int(evaluation_config.get("num_runs", 1)),
            seed=evaluation_config.get("seed"),
        ),
    )

    experience_steps = 0
    collection_device = config["collector"]["device"]
    try:
        with (
            mlflow.start_run(run_name=run_name),
            create_collector(
                algorithm.exploratory_policy,
                collection_device,
                transformed_env,
                config,
            ) as collector,
        ):
            mlflow.log_params(config)
            for batch in collector:
                current_frames = batch.numel()
                flattened_batch = batch.reshape(-1)

                started_at = perf_counter()
                losses = algorithm.learn(flattened_batch)
                elapsed = perf_counter() - started_at
                learning_metrics.push({**losses, "sps": 1.0 / max(elapsed, 1e-9)})

                metric_values = extract_terminal_metric_values(flattened_batch, metrics_spec)
                environment_metrics.push(metric_values)
                final_performance_metrics.push(metric_values)

                collector.update_policy_weights_()
                experience_steps += current_frames
                progress.update(current_frames)
                scheduler.step(current_frames)
                if profiler is not None:
                    profiler.step()

            learning_metrics.flush(step=experience_steps)
            environment_metrics.flush(step=experience_steps)
            if _should_save_final_model(config):
                log_model(algorithm)
    finally:
        progress.close()
        sample_env.close()

    final_metrics = final_performance_metrics.peek()
    if "avg_reward" not in final_metrics:
        raise RuntimeError("An avg_reward metric is required to report training results.")
    return final_metrics["avg_reward"]
