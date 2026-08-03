"""Configuration helpers shared by scripts and MLflow metadata."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("The Sushi Go configuration must be a YAML mapping.")
    return config


def flatten_config(
    config: dict[str, Any], prefix: str = ""
) -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_config(value, name))
        elif value is None:
            flattened[name] = "null"
        elif isinstance(value, (str, int, float, bool)):
            flattened[name] = value
        else:
            flattened[name] = str(value)
    return flattened


def with_preset(config: dict[str, Any], preset: str) -> dict[str, Any]:
    """Return a copy with one league architecture/player-count preset."""

    updated = deepcopy(config)
    environment = updated["environment"]
    model = updated["model"]
    environment["n_players"] = None
    environment["min_n_players"] = None
    environment["max_n_players"] = None
    model["use_encoder"] = False

    if preset.startswith("fixed_"):
        environment["n_players"] = int(preset.removeprefix("fixed_").removesuffix("p"))
    elif preset in {"variable_2_4", "variable_encoder_2_4"}:
        environment["min_n_players"] = 2
        environment["max_n_players"] = 4
        model["use_encoder"] = preset == "variable_encoder_2_4"
    else:
        raise ValueError(f"Unknown Sushi Go preset: {preset}")
    return updated
