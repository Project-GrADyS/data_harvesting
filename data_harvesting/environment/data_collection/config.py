from __future__ import annotations


def evaluation_environment_overrides(config: dict) -> dict:
    """Return data-collection environment overrides used only during evaluation."""
    return {"end_when_all_collected": True}


def requires_masking(config: dict) -> bool:
    """Return whether this environment configuration requires agent masking."""
    env_config = config["environment"]
    return (
        env_config["min_num_agents"] != env_config["max_num_agents"]
        or env_config.get("agent_death_probability", 0.0) > 0.0
    )
