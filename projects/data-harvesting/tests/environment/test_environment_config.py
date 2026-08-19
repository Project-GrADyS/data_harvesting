import pytest

from data_harvesting.environment import evaluation_environment_overrides
from data_harvesting.environment.data_collection.data_collection import (
    DataCollectionEnvironment,
    DataCollectionEnvironmentConfig,
)


def test_evaluation_environment_overrides_force_end_when_all_collected() -> None:
    config = {
        "environment": {
            "sequential_obs": True,
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
        )
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
            DataCollectionEnvironmentConfig(agent_count_sampling="unsupported")
        )


def test_inverse_agent_count_sampling_requires_positive_agent_counts() -> None:
    with pytest.raises(ValueError, match="min_num_agents"):
        DataCollectionEnvironment(
            DataCollectionEnvironmentConfig(
                min_num_agents=0,
                max_num_agents=4,
                agent_count_sampling="inverse",
            )
        )
