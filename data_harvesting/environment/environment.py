import math
import random
from time import sleep
from typing import List, Optional

import gymnasium
import numpy as np
from gradysim.simulator.event import EventLoop
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.interface import INodeHandler
from gradysim.simulator.handler.mobility import MobilityHandler, MobilityConfiguration
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from gradysim.simulator.extension.visualization_controller import VisualizationController
from gradysim.simulator.node import Node
from gradysim.simulator.simulation import SimulationBuilder, Simulator, SimulationConfiguration
from gymnasium.spaces import Box, Dict
from torchrl.envs import EnvBase

from data_harvesting.environment.data_collection import DataCollectionEnvironment, EndCause, GrADySEnvironmentConfig
from data_harvesting.environment.protocols import SensorProtocol, DroneProtocol

def make_env(config: dict) -> EnvBase:
    """
    Create a torchrl-wrapped GrADySEnvironment.
    """
    env_config = config["environment"].copy()
    is_sequential = env_config.pop('sequential_obs')

    # Pass through directly; GrADySEnvironmentConfig handles validation and sampling
    gradys_config = GrADySEnvironmentConfig(**env_config)
    env = DataCollectionEnvironment(gradys_config)

    # If the environment is not sequential, we flatten and concatenate the observation components
    if not is_sequential:
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
            out_key=("agents", "observation"),
            del_keys=False
        ))
    return env
