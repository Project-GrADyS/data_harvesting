import random

import pytest

from data_harvesting.environment.data_collection.death import (
    ScheduledDeathScheduler,
    StochasticDeathScheduler,
)


def test_stochastic_scheduler_probability_zero_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(
        random,
        "random",
        lambda: pytest.fail("probability zero should not consume random values"),
    )

    scheduler = StochasticDeathScheduler(0.0)

    assert scheduler.get_deaths(1, [0, 1, 2]) is None


def test_stochastic_scheduler_preserves_independent_agent_trials(monkeypatch) -> None:
    samples = iter([0.1, 0.9, 0.2])
    monkeypatch.setattr(random, "random", lambda: next(samples))

    scheduler = StochasticDeathScheduler(0.5)

    assert scheduler.get_deaths(1, [2, 4, 6]) == [2, 6]


def test_scheduled_scheduler_only_fires_on_configured_one_based_steps(monkeypatch) -> None:
    monkeypatch.setattr(random, "sample", lambda population, k: population[:k])
    scheduler = ScheduledDeathScheduler([2])

    assert scheduler.get_deaths(1, [0, 1]) is None
    assert scheduler.get_deaths(2, [0, 1]) == [0]
    assert scheduler.get_deaths(3, [0, 1]) is None


def test_repeated_scheduled_timestep_selects_distinct_active_agents(monkeypatch) -> None:
    captured = {}

    def sample(population, k):
        captured["population"] = population
        captured["k"] = k
        return population[-k:]

    monkeypatch.setattr(random, "sample", sample)
    scheduler = ScheduledDeathScheduler([3, 3, 3])

    assert scheduler.get_deaths(3, [0, 2, 5, 7]) == [2, 5, 7]
    assert captured == {"population": [0, 2, 5, 7], "k": 3}


@pytest.mark.parametrize("timesteps", [[0], [-1], [True], [1.5]])
def test_scheduled_scheduler_rejects_invalid_timesteps(timesteps) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        ScheduledDeathScheduler(timesteps)


def test_seeded_scheduled_selection_is_reproducible() -> None:
    scheduler = ScheduledDeathScheduler([1, 1])

    random.seed(42)
    first = scheduler.get_deaths(1, [0, 1, 2, 3])
    scheduler.reset()
    random.seed(42)
    second = scheduler.get_deaths(1, [0, 1, 2, 3])

    assert first == second
