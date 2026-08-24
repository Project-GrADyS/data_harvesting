from __future__ import annotations

from typing import Any

from .data_collection import DataCollectionEnvironmentConfig
from .death import (
    DeathScheduler,
    ScheduledDeathScheduler,
    StochasticDeathScheduler,
    scheduler_requires_masking,
)


def make_death_scheduler(environment_config: dict[str, Any]) -> DeathScheduler:
    """Remove death configuration and construct the environment's sole scheduler.

    ``agent_death_probability`` is accepted only as a compatibility path for
    configurations logged before the unified scheduler configuration was added.
    """
    has_scheduler = "death_scheduler" in environment_config
    has_legacy_probability = "agent_death_probability" in environment_config
    if has_scheduler and has_legacy_probability:
        raise ValueError(
            "Configure either death_scheduler or legacy agent_death_probability, not both."
        )

    if has_legacy_probability:
        return StochasticDeathScheduler(
            environment_config.pop("agent_death_probability")
        )

    raw_scheduler = environment_config.pop("death_scheduler", None)
    if raw_scheduler is None:
        return StochasticDeathScheduler(0.0)
    if not isinstance(raw_scheduler, dict):
        raise TypeError("death_scheduler must be a mapping.")

    scheduler_config = raw_scheduler.copy()
    scheduler_type = scheduler_config.pop("type", None)
    if scheduler_type == "stochastic":
        unexpected = set(scheduler_config) - {"probability"}
        if unexpected:
            raise ValueError(
                f"Unexpected stochastic death scheduler options: {sorted(unexpected)}"
            )
        return StochasticDeathScheduler(scheduler_config.get("probability", 0.0))
    if scheduler_type == "scheduled":
        unexpected = set(scheduler_config) - {"timesteps"}
        if unexpected:
            raise ValueError(
                f"Unexpected scheduled death scheduler options: {sorted(unexpected)}"
            )
        timesteps = scheduler_config.get("timesteps", [])
        if isinstance(timesteps, (str, bytes)) or not isinstance(timesteps, list):
            raise TypeError("Scheduled death timesteps must be a list.")
        return ScheduledDeathScheduler(timesteps)

    raise ValueError(
        "death_scheduler.type must be either 'stochastic' or 'scheduled'."
    )


def evaluation_environment_overrides(config: dict) -> dict:
    """Return data-collection environment overrides used only during evaluation."""
    return {
        "end_when_all_collected": True,
        "agent_count_sampling": "uniform",
    }


def requires_masking(
    config: DataCollectionEnvironmentConfig,
    death_scheduler: DeathScheduler,
) -> bool:
    """Return whether this environment configuration requires agent masking."""
    return (
        config.min_num_agents != config.max_num_agents
        or scheduler_requires_masking(death_scheduler)
    )
