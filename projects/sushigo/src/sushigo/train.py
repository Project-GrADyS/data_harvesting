"""MLflow-tracked Sushi Go DQN training orchestration."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import mlflow
import torch
from rl_core import (
    CollectionMode,
    CollectorConfig,
    ConsoleMetricLogger,
    EvaluationConfig,
    Evaluator,
    MLflowMetricLogger,
    MetricsCollector,
    Scheduler,
    make_collector,
)
from torchrl.envs import check_env_specs

from .algorithm import DQNAlgorithm
from .config import flatten_config
from .environment import make_env
from .metrics import (
    environment_metric_specs,
    extract_batch_metrics,
    extract_terminal_metrics,
    training_metric_specs,
)
from .policy import CHECKPOINT_SCHEMA_VERSION, checkpoint_payload


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config["training"]["device"])
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def _collector_config(
    config: dict[str, Any], device: torch.device
) -> CollectorConfig:
    collector = config["collector"]
    return CollectorConfig(
        mode=(
            CollectionMode.ASYNC
            if bool(collector["async"])
            else CollectionMode.SYNC
        ),
        frames_per_batch=int(collector["frames_per_batch"]),
        total_frames=int(config["training"]["total_frames"]),
        num_workers=int(collector["num_workers"]),
        device=device,
        env_device="cpu",
        policy_device=device,
        storing_device="cpu",
    )


def log_checkpoint(
    policy: torch.nn.Module,
    config: dict[str, Any],
    *,
    artifact_name: str = "checkpoint.pt",
) -> None:
    """Log a source-independent versioned policy payload to the active run."""

    with TemporaryDirectory(prefix="sushigo-checkpoint-") as directory:
        path = Path(directory) / artifact_name
        torch.save(checkpoint_payload(policy, config), path)
        mlflow.log_artifact(path, artifact_path="policy")


def _run_evaluation(
    algorithm: DQNAlgorithm,
    config: dict[str, Any],
    *,
    step: int,
) -> dict[str, float]:
    evaluation = config["evaluation"]
    if not bool(evaluation["enabled"]):
        return {}
    policy = deepcopy(algorithm.policy).to("cpu")
    metrics = MetricsCollector(
        specs=environment_metric_specs()[1:],
        loggers=(
            ConsoleMetricLogger(prefix="eval"),
            MLflowMetricLogger(prefix="eval"),
        ),
    )
    evaluator = Evaluator(
        config=EvaluationConfig(
            num_episodes=int(evaluation["num_episodes"]),
            max_steps=int(evaluation["max_steps"]),
        ),
        env_factory=partial(make_env, config, device="cpu", reward_scale=1.0),
        policy=policy,
        metrics=metrics,
        metric_extractor=extract_terminal_metrics,
    )
    return evaluator.run(step)


def train(
    config: dict[str, Any],
    *,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> float:
    """Train one DQN run and return its aggregate raw final score."""

    device = _device(config)
    run_context = (
        mlflow.start_run(run_name=run_name)
        if mlflow.active_run() is None
        else nullcontext()
    )
    with run_context:
        mlflow.log_params(flatten_config(config))
        mlflow.set_tags(
            {
                "sushigo.kind": "training",
                "sushigo.checkpoint_schema": str(CHECKPOINT_SCHEMA_VERSION),
                **(tags or {}),
            }
        )

        sample_environment = make_env(config, device=device)
        check_env_specs(sample_environment)
        algorithm = DQNAlgorithm(sample_environment, config, device)

        training_metrics = MetricsCollector(
            specs=training_metric_specs(),
            loggers=(
                ConsoleMetricLogger(prefix="train"),
                MLflowMetricLogger(prefix="train"),
            ),
            device=device,
        )
        environment_metrics = MetricsCollector(
            specs=environment_metric_specs(),
            loggers=(
                ConsoleMetricLogger(prefix="env"),
                MLflowMetricLogger(prefix="env"),
            ),
            device=device,
        )
        final_metrics = MetricsCollector(
            specs=environment_metric_specs(),
            device=device,
        )

        metrics_interval = int(config["metrics"]["log_every_frames"])
        checkpoint_interval = int(
            config["checkpoint"]["every_frames"]
        )
        evaluation_interval = int(config["evaluation"]["every_frames"])
        scheduler = Scheduler()
        if metrics_interval > 0:
            scheduler.register(
                "metrics",
                every=metrics_interval,
                callback=lambda step: (
                    training_metrics.flush(step=step),
                    environment_metrics.flush(step=step),
                ),
            )
        if checkpoint_interval > 0:
            scheduler.register(
                "checkpoint",
                every=checkpoint_interval,
                callback=lambda step: log_checkpoint(
                    algorithm.policy,
                    config,
                    artifact_name=f"checkpoint-{step}.pt",
                ),
            )
        if bool(config["evaluation"]["enabled"]) and evaluation_interval > 0:
            scheduler.register(
                "evaluation",
                every=evaluation_interval,
                callback=lambda step: _run_evaluation(
                    algorithm, config, step=step
                ),
            )

        environment_factory = partial(make_env, config, device="cpu")
        experience_steps = 0
        try:
            with make_collector(
                config=_collector_config(config, device),
                env_factory=environment_factory,
                policy=algorithm.exploratory_policy,
            ) as collector:
                for batch in collector:
                    started = perf_counter()
                    learning = algorithm.learn(batch)
                    elapsed = perf_counter() - started
                    frames = batch.numel()
                    training_metrics.push(
                        {
                            **learning,
                            "frames_per_second": frames / max(elapsed, 1e-9),
                        }
                    )
                    extracted = extract_batch_metrics(batch)
                    environment_metrics.push(extracted)
                    final_metrics.push(extracted)

                    collector.update_policy_weights_()
                    experience_steps += frames
                    scheduler.step(increment=frames)
        finally:
            sample_environment.close()

        training_metrics.flush(step=experience_steps)
        environment_metrics.flush(step=experience_steps)
        if bool(config["checkpoint"]["save_final"]):
            log_checkpoint(algorithm.policy, config)

        aggregate = final_metrics.peek()
        score = aggregate.get("mean_final_score")
        if score is None:
            raise RuntimeError(
                "Training completed without a terminal final-score metric."
            )
        mlflow.log_metric("final/mean_final_score", score, step=experience_steps)
        return score
