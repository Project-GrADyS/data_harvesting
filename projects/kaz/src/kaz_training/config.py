from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def get_activation(name: str) -> type[nn.Module]:
    activations = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "LeakyReLU": nn.LeakyReLU}
    try:
        return activations[name]
    except KeyError as error:
        raise ValueError(f"Unsupported activation {name!r}; choose from {sorted(activations)}.") from error


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def flatten_config(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(flatten_config(value, name))
        else:
            flattened[name] = value
    return flattened


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "environment",
        "actor",
        "critic",
        "encoder",
        "training",
        "collector",
        "replay_buffer",
        "optimization",
        "metrics",
        "evaluation",
        "checkpoint",
    }
    missing = required.difference(config)
    if missing:
        raise KeyError(f"Configuration is missing sections: {sorted(missing)}")

    env = config["environment"]
    if int(env["num_archers"]) + int(env["num_knights"]) <= 0:
        raise ValueError("KAZ requires at least one archer or knight.")
    for key in ("max_cycles", "spawn_rate"):
        if int(env[key]) <= 0:
            raise ValueError(f"environment.{key} must be positive.")

    training = config["training"]
    for key in ("total_timesteps", "batch_size"):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive.")
    if int(training["warmup_steps"]) < 0:
        raise ValueError("training.warmup_steps must be non-negative.")
    eps_init = float(training["exploration_epsilon_init"])
    eps_end = float(training["exploration_epsilon_end"])
    if not 0.0 <= eps_end <= eps_init <= 1.0:
        raise ValueError("Exploration epsilon must satisfy 0 <= end <= init <= 1.")

    get_activation(str(config["encoder"]["activation_function"]))
