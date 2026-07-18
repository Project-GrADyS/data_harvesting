from __future__ import annotations

from .data_collection import DataCollectionEnvironmentConfig


def evaluation_environment_overrides(config: dict) -> dict:
    """Return data-collection environment overrides used only during evaluation."""
    return {"end_when_all_collected": True}


def requires_masking(config: DataCollectionEnvironmentConfig) -> bool:
    """Return whether this environment configuration requires agent masking."""
    return (
        config.min_num_agents != config.max_num_agents
        or config.agent_death_probability > 0.0
    )
