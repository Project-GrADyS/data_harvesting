from __future__ import annotations

from copy import deepcopy
from typing import Any

import mlflow
import torch
from mlflow import pytorch as mlflow_pytorch
from mlflow import MlflowClient
from torchrl.envs.utils import ExplorationType, set_exploration_type

from data_harvesting.environment import EndCause, make_env


_METRIC_KEYS = (
    "avg_reward",
    "max_reward",
    "sum_reward",
    "avg_collection_time",
    "episode_duration",
    "completion_time",
    "all_collected",
    "num_collected",
)


def _metric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    data = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "min": float(data.min().item()),
        "max": float(data.max().item()),
    }


def _resolve_model_id_from_run(
    run_id: str,
    *,
    model_name: str = "policy_model",
) -> str:
    client = MlflowClient()
    run = client.get_run(run_id)
    experiment_id = run.info.experiment_id

    models = client.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
    )

    if not models:
        raise ValueError(f"No logged model was found for run '{run_id}'.")

    preferred = [model for model in models if model.name == model_name]
    candidates = preferred if preferred else models
    candidates.sort(key=lambda item: item.creation_timestamp or 0, reverse=True)
    return candidates[0].model_id


def load_policy_from_mlflow_run(
    run_id: str,
    *,
    tracking_uri: str | None = None,
    model_name: str = "policy_model",
):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_id = _resolve_model_id_from_run(run_id, model_name=model_name)
    model_uri = f"models:/{model_id}"
    policy = mlflow_pytorch.load_model(model_uri)
    return policy, model_id


def eval(
    policy,
    config: dict[str, Any],
    num_runs: int,
    *,
    visual: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    if num_runs <= 0:
        raise ValueError("num_runs must be greater than 0")

    eval_config = deepcopy(config)
    env_config = eval_config.setdefault("environment", {})
    env_config["render_mode"] = "visual" if visual else None

    env = make_env(eval_config)

    if hasattr(policy, "eval"):
        policy.eval()

    metric_samples: dict[str, list[float]] = {key: [] for key in _METRIC_KEYS}
    cause_counts = {cause: 0 for cause in EndCause}

    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for run_index in range(num_runs):
            if seed is not None:
                env.set_seed(seed + run_index)

            rollout = env.rollout(
                max_steps=eval_config["environment"]["max_episode_length"],
                policy=policy,
                break_when_any_done=True,
            )
            episode_info = rollout.get(("next", "agents", "info"))[-1, 0]

            for key in _METRIC_KEYS:
                metric_samples[key].append(float(episode_info[key]))

            cause_key = "cause" if "cause" in episode_info.keys() else "end_cause"
            cause_value = int(float(episode_info[cause_key]))
            cause = EndCause(cause_value) if cause_value in [c.value for c in EndCause] else EndCause.NONE
            cause_counts[cause] += 1

    if hasattr(env, "close"):
        env.close()

    end_cause_counts = {cause.name: cause_counts[cause] for cause in EndCause}
    end_cause_rate = {
        cause.name: cause_counts[cause] / num_runs for cause in EndCause
    }

    return {
        "num_runs": num_runs,
        "metrics": {key: _metric_stats(values) for key, values in metric_samples.items()},
        "end_cause_counts": end_cause_counts,
        "end_cause_rate": end_cause_rate,
    }
