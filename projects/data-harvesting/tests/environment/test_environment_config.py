import pytest

from data_harvesting.environment import evaluation_environment_overrides
from data_harvesting.environment.data_collection.data_collection import (
    DataCollectionEnvironment,
    DataCollectionEnvironmentConfig,
)
from data_harvesting.environment.data_collection.config import make_death_scheduler
from data_harvesting.environment.data_collection.death import StochasticDeathScheduler


def test_evaluation_environment_overrides_force_end_when_all_collected() -> None:
    config = {
        "environment": {
            "end_when_all_collected": False,
        }
    }

    overrides = evaluation_environment_overrides(config)

    assert overrides == {
        "end_when_all_collected": True,
        "agent_count_sampling": "uniform",
    }
    assert config["environment"]["end_when_all_collected"] is False


def test_inverse_agent_count_sampling_uses_reciprocal_weights(monkeypatch) -> None:
    captured = {}

    def choose(population, *, weights, k):
        captured["population"] = list(population)
        captured["weights"] = weights
        captured["k"] = k
        return [2]

    monkeypatch.setattr("random.choices", choose)
    env = DataCollectionEnvironment(
        DataCollectionEnvironmentConfig(
            min_num_agents=1,
            max_num_agents=4,
            agent_count_sampling="inverse",
        ),
        death_scheduler=StochasticDeathScheduler(),
    )
    try:
        env.reset(seed=0)
        assert sum(agent.exists for agent in env.episode_agents) == 2
        assert captured == {
            "population": [1, 2, 3, 4],
            "weights": [1.0, 0.5, 1.0 / 3.0, 0.25],
            "k": 1,
        }
    finally:
        env.close()


def test_agent_count_sampling_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="agent_count_sampling"):
        DataCollectionEnvironment(
            DataCollectionEnvironmentConfig(agent_count_sampling="unsupported"),
            death_scheduler=StochasticDeathScheduler(),
        )


def test_inverse_agent_count_sampling_requires_positive_agent_counts() -> None:
    with pytest.raises(ValueError, match="min_num_agents"):
        DataCollectionEnvironment(
            DataCollectionEnvironmentConfig(
                min_num_agents=0,
                max_num_agents=4,
                agent_count_sampling="inverse",
            ),
            death_scheduler=StochasticDeathScheduler(),
        )


def test_make_death_scheduler_parses_unified_stochastic_config() -> None:
    environment_config = {
        "death_scheduler": {"type": "stochastic", "probability": 0.25}
    }

    scheduler = make_death_scheduler(environment_config)

    assert isinstance(scheduler, StochasticDeathScheduler)
    assert scheduler.probability == pytest.approx(0.25)
    assert environment_config == {}


def test_make_death_scheduler_adapts_legacy_probability() -> None:
    environment_config = {"agent_death_probability": 0.0005}

    scheduler = make_death_scheduler(environment_config)

    assert isinstance(scheduler, StochasticDeathScheduler)
    assert scheduler.probability == pytest.approx(0.0005)
    assert environment_config == {}


def test_make_death_scheduler_rejects_unified_and_legacy_config_together() -> None:
    with pytest.raises(ValueError, match="either death_scheduler"):
        make_death_scheduler(
            {
                "death_scheduler": {"type": "stochastic", "probability": 0.1},
                "agent_death_probability": 0.1,
            }
        )
