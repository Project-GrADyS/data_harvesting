from __future__ import annotations

from .data_collection import DataCollectionEnvironmentConfig


def requires_masking(config: DataCollectionEnvironmentConfig) -> bool:
    """Return whether this environment configuration requires agent masking."""
    return config.min_num_agents != config.max_num_agents
