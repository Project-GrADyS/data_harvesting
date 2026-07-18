from __future__ import annotations

from .data_collection.config import (
    evaluation_environment_overrides as _data_collection_evaluation_environment_overrides,
)
from .data_collection.config import requires_masking as _requires_data_collection_masking
from .data_collection.data_collection import DataCollectionEnvironmentConfig


def evaluation_environment_overrides(config: dict) -> dict:
    """Return environment overrides that should apply only during evaluation."""
    return _data_collection_evaluation_environment_overrides(config)


def requires_masking(config: dict) -> bool:
    """Return whether the active environment configuration requires agent masking."""
    env_config = config["environment"].copy()
    env_config.pop("sequential_obs", None)
    return _requires_data_collection_masking(DataCollectionEnvironmentConfig(**env_config))
