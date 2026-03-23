from torchrl.envs import EnvBase

from data_harvesting.encoder.output import ActorOutputKeys
from .data_collection import make_data_collection_env, make_data_collection_output_dict

def make_env(config: dict) -> EnvBase:
    """
    Create the environment based on the provided config.
    """
    return make_data_collection_env(config)

def make_output_dict(config: dict) -> ActorOutputKeys:
    """
    Create the output dictionary for the active environment
    """
    return make_data_collection_output_dict(config)