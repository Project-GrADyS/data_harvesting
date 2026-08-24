from torchrl.envs import EnvBase

from .data_collection import DataCollectionEnvironment, DataCollectionEnvironmentConfig
from .config import make_death_scheduler


def make_data_collection_env(config: dict) -> EnvBase:
    """
    Create a torchrl-wrapped GrADySEnvironment.
    """
    flex_encoder_enabled = bool(config["flex_encoder"]["enabled"])
    env_config = config["environment"].copy()
    # Kept as an ignored compatibility key for legacy configurations.
    env_config.pop("sequential_obs", None)
    death_scheduler = make_death_scheduler(env_config)

    # Pass through directly; GrADySEnvironmentConfig handles validation and sampling
    gradys_config = DataCollectionEnvironmentConfig(**env_config)
    env = DataCollectionEnvironment(gradys_config, death_scheduler=death_scheduler)

    # If the flex_encoder is disabled, we need to flatten and concatenate the observation tensors for torchrl.
    if not flex_encoder_enabled:
        from torchrl.envs.transforms import CatTensors, FlattenObservation
        env = env.append_transform(FlattenObservation(
            first_dim=-2,
            last_dim=-1,
            in_keys=[("agents", "observation", "sensors"), ("agents", "observation", "drones")],
            out_keys=[("agents", "observation_flat", "sensors"), ("agents", "observation_flat", "drones")],
        ))
        # Conditionally include agent_id in the concatenated observation only if present
        include_id = env_config.get("id_on_state", True)
        in_keys = [("agents", "observation_flat", "sensors"), ("agents", "observation_flat", "drones")]
        if include_id:
            in_keys.append(("agents", "observation", "agent_id"))
        env = env.append_transform(CatTensors(
            in_keys=in_keys,
            out_key=("agents", "observation", "flat"),
            del_keys=False
        ))
    return env
