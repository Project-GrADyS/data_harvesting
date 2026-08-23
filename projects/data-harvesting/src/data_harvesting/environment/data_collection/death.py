from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from typing import Protocol


class DeathScheduler(Protocol):
    """Select stable agent slots that should die during an environment step."""

    def reset(self) -> None:
        """Reset episode-specific scheduler state."""

    def get_deaths(
        self,
        timestep: int,
        active_agent_slots: Sequence[int],
    ) -> list[int] | None:
        """Return active slot indices that should die, or ``None`` for no deaths."""


class StochasticDeathScheduler:
    """Independently sample every active agent with a fixed per-step probability."""

    def __init__(self, probability: float = 0.0) -> None:
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise TypeError("Stochastic death probability must be a number.")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Stochastic death probability must be in [0, 1].")
        self.probability = float(probability)

    def reset(self) -> None:
        return None

    def get_deaths(
        self,
        timestep: int,
        active_agent_slots: Sequence[int],
    ) -> list[int] | None:
        del timestep
        if self.probability <= 0.0:
            return None
        deaths = [
            slot
            for slot in active_agent_slots
            if random.random() < self.probability
        ]
        return deaths or None


class ScheduledDeathScheduler:
    """Cause one random active-agent death for each matching timestep entry."""

    def __init__(self, timesteps: Sequence[int]) -> None:
        invalid_timesteps = [
            timestep
            for timestep in timesteps
            if isinstance(timestep, bool)
            or not isinstance(timestep, int)
            or timestep <= 0
        ]
        if invalid_timesteps:
            raise ValueError("Scheduled death timesteps must be positive integers.")
        self.timesteps = tuple(timesteps)
        self._death_counts = Counter(self.timesteps)

    def reset(self) -> None:
        return None

    def get_deaths(
        self,
        timestep: int,
        active_agent_slots: Sequence[int],
    ) -> list[int] | None:
        death_count = min(self._death_counts.get(timestep, 0), len(active_agent_slots))
        if death_count == 0:
            return None
        return random.sample(list(active_agent_slots), k=death_count)


def scheduler_requires_masking(scheduler: DeathScheduler) -> bool:
    """Return whether a scheduler can deactivate agents during an episode."""
    if isinstance(scheduler, StochasticDeathScheduler):
        return scheduler.probability > 0.0
    if isinstance(scheduler, ScheduledDeathScheduler):
        return bool(scheduler.timesteps)
    # Custom schedulers are conservatively assumed to be capable of deaths.
    return True
