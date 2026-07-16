from torchrl.envs import EnvBase

from .data_collection import (
    make_data_collection_env,
    make_data_collection_metrics_spec,
)

def make_env(config: dict) -> EnvBase:
    """
    Create the environment based on the provided config.
    """
    return make_data_collection_env(config)

def make_metrics_spec():
    """
    Create the metrics specification for the active environment.
    """
    return make_data_collection_metrics_spec()
