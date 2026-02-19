import dataclasses
import enum
import enum
import math
import random
from typing import Optional

import numpy as np
import torch
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler, MobilityConfiguration
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from tensordict import TensorDictBase
from torchrl.data import Bounded
from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from data_harvesting.environment.gradys_env import BaseGrADySEnvironment
from data_harvesting.environment.protocols import DroneProtocol, SensorProtocol

@dataclasses.dataclass
class GrADySEnvironmentConfig:
    """Configuration for GrADyS environment (only 'relative' observation mode retained)."""

    render_mode: Optional[str] = None  # "visual" | "console"
    algorithm_iteration_interval: float = 0.5
    # Number of drone agents is samples from [min_num_drones, max_num_drones].
    # To fix the number, set min_num_drones == max_num_drones.
    min_num_drones: int = 1
    max_num_drones: int = 1
    # Number of sensors is always sampled each reset from [min_num_sensors, max_num_sensors].
    # To fix the number, set min_num_sensors == max_num_sensors.
    min_num_sensors: int = 2
    max_num_sensors: int = 2
    scenario_size: float = 100
    max_episode_length: int = 500
    max_seconds_stalled: int = 30
    communication_range: float = 20
    state_num_closest_sensors: int = 2
    state_num_closest_drones: int = 2
    id_on_state: bool = True
    min_sensor_priority: float = 0.1
    max_sensor_priority: float = 1
    full_random_drone_position: bool = False
    reward: str = 'punish'  # Fixed reward mode: punish
    speed_action: bool = True
    end_when_all_collected: bool = True

class EndCause(enum.Enum):
    NONE = 0
    TIMEOUT = 1
    ALL_COLLECTED = 2
    STALLED = 3

class DataCollectionEnvironment(BaseGrADySEnvironment, EnvBase):
    """
    A specialized environment for data collection in simulations, extending the GrADySEnvironment.
    This environment simulates sensor data collection with autonomous agents.
    """

    _simulation_configuration: SimulationConfiguration

    batch_locked: bool = True

    def __init__(self, config: GrADySEnvironmentConfig, *, device=None):
        BaseGrADySEnvironment.__init__(self, config.algorithm_iteration_interval, visual_mode=(config.render_mode == "visual"))
        EnvBase.__init__(self, device=device)

        self.render_mode = config.render_mode
        self.algorithm_iteration_interval = config.algorithm_iteration_interval

        if config.min_num_sensors > config.max_num_sensors:
            raise ValueError("min_num_sensors cannot be greater than max_num_sensors.")

        self.min_num_sensors = config.min_num_sensors
        self.max_num_sensors = config.max_num_sensors
        self.min_num_drones = config.min_num_drones
        self.max_num_drones = config.max_num_drones

        self.scenario_size = config.scenario_size
        self.max_episode_length = config.max_episode_length
        self.max_seconds_stalled = config.max_seconds_stalled
        self.communication_range = config.communication_range
        self.state_num_closest_sensors = config.state_num_closest_sensors
        self.state_num_closest_drones = config.state_num_closest_drones
        self.id_on_state = config.id_on_state
        self.min_sensor_priority = config.min_sensor_priority
        self.max_sensor_priority = config.max_sensor_priority
        self.full_random_drone_position = config.full_random_drone_position
        if config.reward != "punish":
            raise ValueError("Only reward='punish' is supported.")
        self.speed_action = config.speed_action
        self.end_when_all_collected = config.end_when_all_collected

        self.possible_agents = [f"drone{i}" for i in range(self.max_num_drones)]
        self.group_map = {"agents": self.possible_agents}

        self.active_num_sensors: int = -1
        self.active_num_drones: int = -1
        self.agents: list[str] = []
        self.episode_duration = 0
        self.stall_duration = 0
        self.reward_sum = 0.0
        self.max_reward = -math.inf
        self.sensors_collected = 0
        self.collection_times: list[float] = []

        self._info_keys = [
            "avg_reward",
            "max_reward",
            "sum_reward",
            "avg_collection_time",
            "episode_duration",
            "completion_time",
            "all_collected",
            "num_collected",
            "cause",
        ]

        self._simulation_configuration = SimulationConfiguration(
            debug=False,
            execution_logging=False,
            duration=self.max_episode_length,
        )

        self._build_specs()
        self._cached_reset_zero = self.full_observation_spec.zero()
        self._cached_reset_zero.update(self.full_done_spec.zero())

        self._cached_step_zero = self.full_observation_spec.zero()
        self._cached_step_zero.update(self.full_reward_spec.zero())
        self._cached_step_zero.update(self.full_done_spec.zero())

    def _build_simulation(self, builder: SimulationBuilder):
        """
        Set up the GrADyS-SIM NextGen simulation environment with the provided configuration.

        Args:
            builder (SimulationBuilder): Builder object for setting up the simulation.
        """
        # Adding necessary handlers to the simulation builder
        builder.add_handler(CommunicationHandler(CommunicationMedium(
            transmission_range=self.communication_range
        )))
        builder.add_handler(MobilityHandler(MobilityConfiguration(
            update_rate=self.algorithm_iteration_interval / 2
        )))
        builder.add_handler(TimerHandler())
        if self.render_mode == "visual":
            builder.add_handler(VisualizationHandler(VisualizationConfiguration(
                open_browser=False,
                x_range=(-self.scenario_size, self.scenario_size),
                y_range=(-self.scenario_size, self.scenario_size),
                z_range=(0, self.scenario_size),
            )))


        self.sensor_node_ids = []
        SensorProtocol.min_priority = self.min_sensor_priority
        SensorProtocol.max_priority = self.max_sensor_priority

        for i in range(self.active_num_sensors):
            self.sensor_node_ids.append(builder.add_node(SensorProtocol, (
                random.uniform(-self.scenario_size, self.scenario_size),
                random.uniform(-self.scenario_size, self.scenario_size),
                0
            )))

        self.agent_node_ids = []
        DroneProtocol.speed_action = self.speed_action
        DroneProtocol.algorithm_interval = self.algorithm_iteration_interval

        for _ in range(self.active_num_drones):
            if self.full_random_drone_position:
                self.agent_node_ids.append(builder.add_node(DroneProtocol, (
                    random.uniform(-self.scenario_size, self.scenario_size),
                    random.uniform(-self.scenario_size, self.scenario_size),
                    0
                )))
            else:
                self.agent_node_ids.append(builder.add_node(DroneProtocol, (
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    0
                )))

        self.simulator = builder.build()

    def _build_specs(self) -> None:
        device = self.device

        action_dim = 2 if self.speed_action else 1
        sensors_shape = (self.max_num_drones, self.state_num_closest_sensors, 2)
        drones_shape = (self.max_num_drones, self.state_num_closest_drones, 2)
        agent_id_shape = (self.max_num_drones, 1)
        mask_shape = (self.max_num_drones,)
        action_shape = (self.max_num_drones, action_dim)
        reward_shape = (self.max_num_drones, 1)
        done_shape = (self.max_num_drones, 1)

        obs_inner = {
            "sensors": Bounded(
                -torch.ones(sensors_shape, device=device),
                torch.ones(sensors_shape, device=device),
                sensors_shape,
                dtype=torch.float32,
                device=device,
            ),
            "drones": Bounded(
                -torch.ones(drones_shape, device=device),
                torch.ones(drones_shape, device=device),
                drones_shape,
                dtype=torch.float32,
                device=device,
            ),
        }
        if self.id_on_state:
            obs_inner["agent_id"] = Bounded(
                torch.zeros(agent_id_shape, device=device),
                torch.ones(agent_id_shape, device=device),
                agent_id_shape,
                dtype=torch.float32,
                device=device,
            )

        observation_spec = Composite(
            {
                "agents": Composite(
                    {
                        "observation": Composite(obs_inner, device=device),
                        "mask": Categorical(
                            n=2,
                            shape=mask_shape,
                            dtype=torch.bool,
                            device=device,
                        ),
                        "info": Composite(
                            {
                                key: Unbounded(
                                    shape=(self.max_num_drones,),
                                    device=device,
                                    dtype=torch.float32,
                                )
                                for key in self._info_keys
                            },
                            device=device,
                        ),
                    },
                    device=device,
                )
            },
            device=device,
        )

        action_spec = Composite(
            {
                "agents": Composite(
                    {
                        "action": Bounded(
                            torch.zeros(action_shape, device=device),
                            torch.ones(action_shape, device=device),
                            action_shape,
                            dtype=torch.float32,
                            device=device,
                        )
                    },
                    device=device,
                )
            },
            device=device,
        )

        reward_spec = Composite(
            {
                "agents": Composite(
                    {
                        "reward": Unbounded(
                            shape=reward_shape, device=device, dtype=torch.float32
                        )
                    },
                    device=device,
                )
            },
            device=device,
        )

        done_spec = Composite(
            {
                "done": Categorical(
                    n=2, shape=(1,), dtype=torch.bool, device=device
                ),
                "terminated": Categorical(
                    n=2, shape=(1,), dtype=torch.bool, device=device
                ),
                "truncated": Categorical(
                    n=2, shape=(1,), dtype=torch.bool, device=device
                ),
                "agents": Composite(
                    {
                        "done": Categorical(
                            n=2, shape=done_shape, dtype=torch.bool, device=device
                        ),
                        "terminated": Categorical(
                            n=2, shape=done_shape, dtype=torch.bool, device=device
                        ),
                        "truncated": Categorical(
                            n=2, shape=done_shape, dtype=torch.bool, device=device
                        ),
                    },
                    device=device,
                ),
            },
            device=device,
        )

        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.done_spec = done_spec

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        actions = tensordict.get(("agents", "action"))
        self._apply_actions(actions)

        # We record the collected sensors before stepping the simulation so we can compare with
        # that figure after the step to see if any new sensors were collected
        collected_before = self._get_sensor_collected()

        status = self.step_simulation()
        end_cause = EndCause.NONE
        if status.has_ended:
            end_cause = EndCause.TIMEOUT

        collected_after = self._get_sensor_collected()

        # Update the stall counter based on whether new sensors were collected
        # A stall is when the agents fail to collect any new sensors. If the stall
        # persists for too long, the episode ends.
        self._update_stall(collected_before, collected_after)

        # Check if all sensors have been collected
        all_sensors_collected = sum(collected_after) == self.active_num_sensors

        if self.stall_duration > self.max_seconds_stalled:
            end_cause = EndCause.STALLED
        if all_sensors_collected:
            end_cause = EndCause.ALL_COLLECTED

        reward = self._compute_reward(collected_before, collected_after)
        self._update_collection_times(collected_after)
        self._reward_sum_update(reward)

        # We do not end the simulation when all sensors are collected unless self.end_when_all_collected is True. We've found that training
        # benefits from time after collection where agents can "enjoy" the reward signal for success.
        simulation_ended = (
            (all_sensors_collected and self.end_when_all_collected)
            or status.has_ended
            or end_cause in {EndCause.STALLED, EndCause.TIMEOUT}
        )

        # Filling the output tensordict for the step
        tensordict_out = self._cached_step_zero.clone()
        self._fill_observation(tensordict_out, self._observe_simulation())
        self._fill_rewards(tensordict_out, reward)
        self._fill_done(tensordict_out, end_cause)
        self._fill_info(tensordict_out, sum(collected_after), end_cause, simulation_ended)
        return tensordict_out

    def _reset_statistics(self):
        """
        Resets the statistics for a new episode.
        """
        self.episode_duration = 0
        self.stall_duration = 0
        self.reward_sum = 0
        self.max_reward = -math.inf
        self.sensors_collected = 0
        self.collection_times = [self.max_episode_length for _ in range(self.active_num_sensors)]

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        # Picking number of sensors and drones for this episode
        self.active_num_sensors = random.randint(self.min_num_sensors, self.max_num_sensors)
        self.active_num_drones = random.randint(self.min_num_drones, self.max_num_drones)
        self.agents = [f"drone{i}" for i in range(self.active_num_drones)]

        self._reset_statistics()

        self.reset_simulation(self._simulation_configuration)

        max_ready_steps = self.active_num_drones * 10  # Arbitrary large number of steps to wait for drones to be ready
        ready_steps = 0
        while not self._all_active_drones_ready():
            status = self.step_simulation()
            ready_steps += 1
            if status.has_ended:
                raise RuntimeError("Simulation ended before all drones received initial telemetry")
            if ready_steps >= max_ready_steps:
                raise RuntimeError("Timed out waiting for initial telemetry for all drones")
        
        all_obs = self._observe_simulation()

        # The initial observation has to contain observations for all possible agents
        # We repeat the last active agent's observation for the inactive agents
        # This is not a problem because these agents will be truncated immediately
        for i in range(self.active_num_drones, self.max_num_drones):
            all_obs[f"drone{i}"] = all_obs[f"drone{self.active_num_drones - 1}"]
        tensordict_out = self._cached_reset_zero.clone()
        self._fill_observation(tensordict_out, all_obs)
        self._fill_done(tensordict_out, EndCause.NONE)
        self._fill_info(tensordict_out, 0, EndCause.NONE, False)
        return tensordict_out

    def _all_active_drones_ready(self) -> bool:
        for index in range(self.active_num_drones):
            agent_node = self.simulator.get_node(self.agent_node_ids[index])
            protocol = agent_node.protocol_encapsulator.protocol
            if not getattr(protocol, "ready", False):
                return False
        return True
    
    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.reset(seed=seed)

    def _apply_actions(self, actions: torch.Tensor) -> None:
        if self.active_num_drones <= 0:
            return

        for index in range(self.active_num_drones):
            agent_node = self.simulator.get_node(self.agent_node_ids[index])
            action = actions[index].detach().cpu().tolist()
            agent_node.protocol_encapsulator.protocol.act(action, self.scenario_size)

    def _get_sensor_collected(self) -> list[bool]:
        return [
            self.simulator.get_node(sensor_id).protocol_encapsulator.protocol.has_collected
            for sensor_id in self.sensor_node_ids
        ]

    def _update_stall(self, collected_before: list[bool], collected_after: list[bool]) -> None:
        if sum(collected_after) > sum(collected_before):
            self.stall_duration = 0
        else:
            self.stall_duration += self.algorithm_iteration_interval

        self.episode_duration += 1

    def _compute_reward(self, collected_before: list[bool], collected_after: list[bool]) -> float:
        before = sum(collected_before)
        after = sum(collected_after)
        if after > before:
            return float((after - before) * 10)
        remaining = self.active_num_sensors - after
        return float(-(remaining) / max(1, self.active_num_sensors))

    def _update_collection_times(self, collected_after: list[bool]) -> None:
        current_timestamp = self.episode_duration * self.algorithm_iteration_interval
        for index, sensor_id in enumerate(self.sensor_node_ids):
            if collected_after[index] and self.collection_times[index] == self.max_episode_length:
                self.collection_times[index] = current_timestamp

    def _reward_sum_update(self, reward: float) -> None:
        if self.active_num_drones > 0:
            self.reward_sum += reward
            self.max_reward = max(self.max_reward, reward)

    def _observe_simulation(self) -> dict:
        """
        Extracts information from the simulation to form the observation for each agent. 
        Each agent's observation includes the positions of the closest unvisited sensors
        and the closest other drones, normalized within the scenario size.
        Returns:
            dict: A dictionary containing observations for each agent.
        """
        sensor_nodes = np.array([
            self.simulator.get_node(sensor_id).position[:2]
            for sensor_id in self.sensor_node_ids
        ])
        unvisited_sensor_mask = np.array([
            not self.simulator.get_node(sensor_id)
            .protocol_encapsulator.protocol.has_collected
            for sensor_id in self.sensor_node_ids
        ])
        unvisited_sensor_nodes = sensor_nodes[unvisited_sensor_mask]

        agent_nodes = np.array([
            self.simulator.get_node(agent_id).position[:2]
            for agent_id in self.agent_node_ids
        ])

        max_distance = self.scenario_size * 2

        state = {}
        for agent_index in range(self.active_num_drones):
            agent_position = agent_nodes[agent_index]

            sensor_distances = np.linalg.norm(
                unvisited_sensor_nodes - agent_position, axis=1
            ) if len(unvisited_sensor_nodes) else np.array([])
            sorted_sensor_indices = np.argsort(sensor_distances)

            closest_unvisited_sensors = np.zeros((self.state_num_closest_sensors, 2))
            closest_unvisited_sensors.fill(-1)
            if len(sorted_sensor_indices):
                closest_unvisited_sensors[:len(sorted_sensor_indices)] = unvisited_sensor_nodes[
                    sorted_sensor_indices[:self.state_num_closest_sensors]
                ]

            agent_distances = np.linalg.norm(agent_nodes - agent_position, axis=1)
            sorted_agent_indices = np.argsort(agent_distances)

            closest_agents = np.zeros((self.state_num_closest_drones, 2))
            closest_agents.fill(-1)
            if len(sorted_agent_indices) > 1:
                closest_agents[:len(sorted_agent_indices) - 1] = agent_nodes[
                    sorted_agent_indices[1:self.state_num_closest_drones + 1]
                ]

            if len(sorted_agent_indices) > 1:
                closest_agents[:len(sorted_agent_indices) - 1] = (
                    closest_agents[:len(sorted_agent_indices) - 1] - agent_position + max_distance
                ) / (max_distance * 2)

            if len(sorted_sensor_indices):
                closest_unvisited_sensors[:len(sorted_sensor_indices)] = (
                    closest_unvisited_sensors[:len(sorted_sensor_indices)] - agent_position + max_distance
                ) / (max_distance * 2)

            state[f"drone{agent_index}"] = {
                "drones": closest_agents,
                "sensors": closest_unvisited_sensors,
            }
            if self.id_on_state:
                state[f"drone{agent_index}"]["agent_id"] = np.array([
                    agent_index / (self.max_num_drones - 1) if self.max_num_drones > 1 else 0
                ])
        return state

    def _blank_info(self) -> dict:
        return {
            agent: {
                "avg_reward": 0.0,
                "max_reward": 0.0,
                "sum_reward": 0.0,
                "avg_collection_time": 0.0,
                "episode_duration": 0.0,
                "completion_time": 0.0,
                "all_collected": 0.0,
                "num_collected": 0.0,
                "cause": EndCause.NONE.value,
            }
            for agent in self.possible_agents
        }

    def _fill_observation(
        self,
        td: TensorDictBase,
        observation_dict: dict,
    ) -> None:
        """
        Fills the input tensordict with observations from the simulation. 
        Args:
            td (TensorDictBase): The tensordict to fill with observations.
            observation_dict (Optional[dict]): Precomputed observations.
        """
        agents_td = td.get("agents")
        obs_td = agents_td.get("observation")

        sensors = obs_td.get("sensors")
        drones = obs_td.get("drones")
        sensors.zero_()
        drones.zero_()

        if self.id_on_state:
            agent_id = obs_td.get("agent_id")
            agent_id.zero_()

        mask = agents_td.get("mask")
        mask.zero_()

        for i in range(self.active_num_drones):
            agent_name = f"drone{i}"
            obs = observation_dict[agent_name]
            sensors[i].copy_(torch.as_tensor(obs["sensors"], device=self.device))
            drones[i].copy_(torch.as_tensor(obs["drones"], device=self.device))
            if self.id_on_state:
                agent_id[i].copy_(
                    torch.as_tensor(obs["agent_id"], device=self.device)
                )
            mask[i] = True

    def _fill_rewards(self, td: TensorDictBase, reward_value: float) -> None:
        reward = td.get(("agents", "reward"))
        reward.zero_()
        if self.active_num_drones > 0:
            reward[: self.active_num_drones, 0] = reward_value

    def _fill_done(self, td: TensorDictBase, end_cause: EndCause) -> None:
        done = td.get(("agents", "done"))
        terminated = td.get(("agents", "terminated"))
        truncated = td.get(("agents", "truncated"))
        done.zero_()
        terminated.zero_()
        truncated.zero_()

        active = self.active_num_drones
        if end_cause != EndCause.NONE:
            terminated[:active, 0] = True
            done[:active, 0] = True

        if active < self.max_num_drones:
            truncated[active:, 0] = True
            done[active:, 0] = True

        if active > 0:
            done_any = bool(done[:active].any())
            terminated_any = bool(terminated[:active].any())
            truncated_any = bool(truncated[:active].any())
        else:
            done_any = False
            terminated_any = False
            truncated_any = False

        td.set("done", torch.tensor([done_any], device=self.device))
        td.set("terminated", torch.tensor([terminated_any], device=self.device))
        td.set("truncated", torch.tensor([truncated_any], device=self.device))

    def _fill_info(self, td: TensorDictBase, num_collected: int, end_cause: EndCause, ended: bool) -> None:
        all_collected = num_collected == self.active_num_sensors
        info_td = td.get(("agents", "info"))
        for key in self._info_keys:
            info_td.get(key).zero_()

        if not ended or self.active_num_drones == 0:
            return

        avg_reward = self.reward_sum / max(1, self.episode_duration)
        avg_collection_time = (
            sum(self.collection_times) / self.active_num_sensors
            if self.active_num_sensors > 0
            else 0.0
        )
        completion_time = (
            self.max_episode_length
            if not all_collected
            else self.simulator._current_timestamp
        )

        metrics = {
            "avg_reward": avg_reward,
            "max_reward": self.max_reward,
            "sum_reward": self.reward_sum,
            "avg_collection_time": avg_collection_time,
            "episode_duration": float(self.episode_duration),
            "completion_time": float(completion_time),
            "all_collected": float(int(all_collected)),
            "num_collected": float(num_collected),
            "cause": float(end_cause.value),
        }

        for key, value in metrics.items():
            info_td.get(key)[: self.active_num_drones] = value

    def close(self, *, raise_if_closed: bool = True):
        super().close(raise_if_closed=raise_if_closed)
        self.finalize_simulation()