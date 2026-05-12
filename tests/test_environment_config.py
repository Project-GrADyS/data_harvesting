from data_harvesting.environment import evaluation_environment_overrides


def test_evaluation_environment_overrides_force_end_when_all_collected() -> None:
    config = {
        "environment": {
            "sequential_obs": True,
            "end_when_all_collected": False,
        }
    }

    overrides = evaluation_environment_overrides(config)

    assert overrides == {"end_when_all_collected": True}
    assert config["environment"]["end_when_all_collected"] is False
