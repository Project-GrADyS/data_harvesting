from __future__ import annotations

import random

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.data import Bounded
from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.envs.common import EnvBase

from data_harvesting.environment import EndCause, GrADySEnvironmentConfig


class GradysTorchrlEnvironment(EnvBase):
    """Torch-native GrADyS environment (no PettingZoo wrapper)."""

    batch_locked: bool = True

    def __init__(self, config: GrADySEnvironmentConfig, *, device=None):
        super().__init__(device=device)
        self.config = config

        self.min_num_drones = config.min_num_drones
        self.max_num_drones = config.max_num_drones
        self.min_num_sensors = config.min_num_sensors
        self.max_num_sensors = config.max_num_sensors

        self.scenario_size = float(config.scenario_size)
        self.max_episode_length = int(config.max_episode_length)
        self.max_seconds_stalled = float(config.max_seconds_stalled)
        self.communication_range = float(config.communication_range)
        self.state_num_closest_sensors = int(config.state_num_closest_sensors)
        self.state_num_closest_drones = int(config.state_num_closest_drones)
        self.id_on_state = bool(config.id_on_state)
        self.min_sensor_priority = float(config.min_sensor_priority)
        self.max_sensor_priority = float(config.max_sensor_priority)
        self.full_random_drone_position = bool(config.full_random_drone_position)
        self.reward_type = str(config.reward)
        self.speed_action = bool(config.speed_action)
        self.end_when_all_collected = bool(config.end_when_all_collected)
        self.algorithm_iteration_interval = float(config.algorithm_iteration_interval)

        self.action_dim = 2 if self.speed_action else 1
        self._agent_names = [f"drone{i}" for i in range(self.max_num_drones)]
        self.group_map = {"agents": self._agent_names}

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

        self._build_specs()
        self._init_state_tensors()

        self._cached_reset_zero = self.full_observation_spec.zero()
        self._cached_reset_zero.update(self.full_done_spec.zero())

        self._cached_step_zero = self.full_observation_spec.zero()
        self._cached_step_zero.update(self.full_reward_spec.zero())
        self._cached_step_zero.update(self.full_done_spec.zero())

    def _build_specs(self) -> None:
        device = self.device

        sensors_shape = (self.max_num_drones, self.state_num_closest_sensors, 2)
        drones_shape = (self.max_num_drones, self.state_num_closest_drones, 2)
        agent_id_shape = (self.max_num_drones, 1)
        mask_shape = (self.max_num_drones,)
        action_shape = (self.max_num_drones, self.action_dim)
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

    def _init_state_tensors(self) -> None:
        device = self.device
        self._active_num_drones = 0
        self._active_num_sensors = 0

        self._drone_positions = torch.zeros(
            (self.max_num_drones, 2), device=device, dtype=torch.float32
        )
        self._sensor_positions = torch.zeros(
            (self.max_num_sensors, 2), device=device, dtype=torch.float32
        )
        self._sensor_priorities = torch.zeros(
            (self.max_num_sensors,), device=device, dtype=torch.float32
        )
        self._sensor_collected = torch.zeros(
            (self.max_num_sensors,), device=device, dtype=torch.bool
        )
        self._sensor_active = torch.zeros(
            (self.max_num_sensors,), device=device, dtype=torch.bool
        )

        self._episode_duration = 0
        self._stall_duration = 0.0
        self._reward_sum = 0.0
        self._max_reward = -float("inf")
        self._collection_times = torch.zeros(
            (self.max_num_sensors,), device=device, dtype=torch.float32
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.reset(seed=seed)

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        _ = tensordict
        self._active_num_drones = random.randint(
            self.min_num_drones, self.max_num_drones
        )
        self._active_num_sensors = random.randint(
            self.min_num_sensors, self.max_num_sensors
        )

        self._sensor_active.zero_()
        self._sensor_active[: self._active_num_sensors] = True
        self._sensor_collected.zero_()

        self._sensor_priorities.uniform_(
            self.min_sensor_priority, self.max_sensor_priority
        )

        self._sensor_positions.uniform_(
            -self.scenario_size, self.scenario_size
        )
        self._drone_positions.zero_()
        if self.full_random_drone_position:
            self._drone_positions[: self._active_num_drones].uniform_(
                -self.scenario_size, self.scenario_size
            )
        else:
            self._drone_positions[: self._active_num_drones].uniform_(-2.0, 2.0)

        self._episode_duration = 0
        self._stall_duration = 0.0
        self._reward_sum = 0.0
        self._max_reward = -float("inf")
        self._collection_times.fill_(float(self.max_episode_length))

        tensordict_out = self._cached_reset_zero.clone()
        self._fill_observation(tensordict_out)
        self._fill_done(tensordict_out, end_cause=EndCause.NONE)
        return tensordict_out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        actions = tensordict.get(("agents", "action"))
        self._apply_actions(actions)

        collected_before = self._sensor_collected.clone()
        self._update_collections()

        reward = self._compute_reward(collected_before)
        self._reward_sum += reward
        self._max_reward = max(self._max_reward, reward)

        if int(self._sensor_collected.sum().item()) > int(collected_before.sum().item()):
            self._stall_duration = 0.0
        else:
            self._stall_duration += self.algorithm_iteration_interval

        self._episode_duration += 1

        all_sensors_collected = (
            int(self._sensor_collected.sum().item()) == self._active_num_sensors
        )

        end_cause = EndCause.NONE
        if all_sensors_collected:
            end_cause = EndCause.ALL_COLLECTED
        if self._stall_duration > self.max_seconds_stalled:
            end_cause = EndCause.STALLED
        if self._episode_duration >= self.max_episode_length:
            end_cause = EndCause.TIMEOUT

        simulation_ended = (
            (all_sensors_collected and self.end_when_all_collected)
            or end_cause in {EndCause.TIMEOUT, EndCause.STALLED}
        )

        tensordict_out = self._cached_step_zero.clone()
        self._fill_observation(tensordict_out, end_cause=end_cause, ended=simulation_ended)
        self._fill_rewards(tensordict_out, reward)
        self._fill_done(tensordict_out, end_cause=end_cause)
        return tensordict_out

    def _apply_actions(self, actions: torch.Tensor) -> None:
        if self._active_num_drones == 0:
            return

        active_actions = actions[: self._active_num_drones]
        direction = active_actions[:, 0] * (2 * torch.pi)
        if self.speed_action:
            speed = active_actions[:, 1] * 15.0
        else:
            speed = torch.ones_like(direction)

        distance = speed * self.algorithm_iteration_interval
        delta = torch.stack((torch.cos(direction), torch.sin(direction)), dim=-1)
        movement = delta * distance.unsqueeze(-1)
        self._drone_positions[: self._active_num_drones] += movement

        self._drone_positions[: self._active_num_drones].clamp_(
            -self.scenario_size, self.scenario_size
        )

    def _update_collections(self) -> None:
        if self._active_num_drones == 0 or self._active_num_sensors == 0:
            return

        drones = self._drone_positions[: self._active_num_drones]
        sensors = self._sensor_positions[: self._active_num_sensors]
        distances = torch.cdist(drones, sensors)
        collected = distances <= self.communication_range
        newly_collected = collected.any(dim=0)
        self._sensor_collected[: self._active_num_sensors] |= newly_collected

        current_time = self._episode_duration * self.algorithm_iteration_interval
        first_time = (
            newly_collected
            & (self._collection_times[: self._active_num_sensors] == self.max_episode_length)
        )
        self._collection_times[: self._active_num_sensors][first_time] = current_time

    def _compute_reward(self, collected_before: torch.Tensor) -> float:
        collected_after = self._sensor_collected
        before = int(collected_before.sum().item())
        after = int(collected_after.sum().item())

        reward = 0.0
        if self.reward_type == "punish":
            if after > before:
                reward = float(after - before) * 10.0
            else:
                remaining = self._active_num_sensors - after
                reward = -(remaining) / max(1, self._active_num_sensors)
        elif self.reward_type == "reward":
            reward = float(after - before)

        if self.reward_type == "time-reward":
            newly_collected = collected_after & (~collected_before)
            current_time = self._episode_duration * self.algorithm_iteration_interval
            if newly_collected.any():
                priorities = self._sensor_priorities[: self._active_num_sensors]
                newly = newly_collected[: self._active_num_sensors]
                reward += float(
                    (priorities[newly] * (1 - current_time / self.max_episode_length)).sum()
                )

        return reward

    def _fill_observation(
        self,
        td: TensorDictBase,
        *,
        end_cause: EndCause = EndCause.NONE,
        ended: bool = False,
    ) -> None:
        agents_td = td.get("agents")
        obs_td = agents_td.get("observation")

        sensors = obs_td.get("sensors")
        drones = obs_td.get("drones")
        sensors.fill_(-1)
        drones.fill_(-1)

        if self.id_on_state:
            agent_id = obs_td.get("agent_id")
            agent_id.zero_()

        mask = agents_td.get("mask")
        mask.zero_()

        if self._active_num_drones == 0:
            self._fill_info(agents_td.get("info"), end_cause=end_cause, ended=ended)
            return

        active_drone_pos = self._drone_positions[: self._active_num_drones]
        active_sensor_pos = self._sensor_positions[: self._active_num_sensors]
        active_sensor_collected = self._sensor_collected[: self._active_num_sensors]

        max_distance = self.scenario_size * 2
        large = max_distance * 4

        if self._active_num_sensors > 0:
            distances = torch.cdist(active_drone_pos, active_sensor_pos)
            distances = distances.clone()
            distances[:, active_sensor_collected] = large
            topk = min(self.state_num_closest_sensors, self._active_num_sensors)
            values, indices = torch.topk(
                distances, k=topk, dim=1, largest=False
            )
            selected = active_sensor_pos[indices]
            normalized = (active_drone_pos[:, None, :] - selected + max_distance) / (
                max_distance * 2
            )
            sensors[: self._active_num_drones, :topk] = normalized
            invalid = values >= large / 2
            if invalid.any():
                sensors[: self._active_num_drones, :topk][invalid] = -1

        if self._active_num_drones > 1:
            drone_dist = torch.cdist(active_drone_pos, active_drone_pos)
            diag = torch.eye(self._active_num_drones, device=self.device) * large
            drone_dist = drone_dist + diag
            topk = min(self.state_num_closest_drones, self._active_num_drones - 1)
            if topk > 0:
                values, indices = torch.topk(
                    drone_dist, k=topk, dim=1, largest=False
                )
                selected = active_drone_pos[indices]
                normalized = (active_drone_pos[:, None, :] - selected + max_distance) / (
                    max_distance * 2
                )
                drones[: self._active_num_drones, :topk] = normalized
                invalid = values >= large / 2
                if invalid.any():
                    drones[: self._active_num_drones, :topk][invalid] = -1

        if self.id_on_state:
            if self.max_num_drones > 1:
                ids = (
                    torch.arange(self.max_num_drones, device=self.device)
                    / (self.max_num_drones - 1)
                ).unsqueeze(-1)
            else:
                ids = torch.zeros((self.max_num_drones, 1), device=self.device)
            agent_id.copy_(ids)

        mask[: self._active_num_drones] = True

        self._fill_info(agents_td.get("info"), end_cause=end_cause, ended=ended)

    def _fill_info(self, info_td, *, end_cause: EndCause, ended: bool) -> None:
        for key in self._info_keys:
            info_td.get(key).zero_()

        if not ended:
            return

        active = self._active_num_drones
        if active == 0:
            return

        avg_reward = self._reward_sum / max(1, self._episode_duration)
        avg_collection_time = float(
            self._collection_times[: self._active_num_sensors].mean().item()
            if self._active_num_sensors > 0
            else 0.0
        )
        completion_time = (
            self._episode_duration * self.algorithm_iteration_interval
            if end_cause != EndCause.ALL_COLLECTED
            else float(
                self._collection_times[: self._active_num_sensors].max().item()
                if self._active_num_sensors > 0
                else 0.0
            )
        )

        metrics = {
            "avg_reward": avg_reward,
            "max_reward": self._max_reward,
            "sum_reward": self._reward_sum,
            "avg_collection_time": avg_collection_time,
            "episode_duration": float(self._episode_duration),
            "completion_time": completion_time,
            "all_collected": float(
                int(
                    self._sensor_collected[: self._active_num_sensors].all().item()
                )
            ),
            "num_collected": float(
                int(self._sensor_collected[: self._active_num_sensors].sum().item())
            ),
            "cause": float(end_cause.value),
        }

        for key, value in metrics.items():
            info_td.get(key)[:active] = value

    def _fill_rewards(self, td: TensorDictBase, reward_value: float) -> None:
        reward = td.get(("agents", "reward"))
        reward.zero_()
        if self._active_num_drones:
            reward[: self._active_num_drones, 0] = reward_value

    def _fill_done(self, td: TensorDictBase, end_cause: EndCause) -> None:
        done = td.get(("agents", "done"))
        terminated = td.get(("agents", "terminated"))
        truncated = td.get(("agents", "truncated"))
        done.zero_()
        terminated.zero_()
        truncated.zero_()

        active = self._active_num_drones
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

    def close(self, *, raise_if_closed: bool = True) -> None:
        _ = raise_if_closed


def make_torchrl_env(config: dict) -> EnvBase:
    env_config = config["environment"].copy()
    is_sequential = env_config.pop("sequential_obs")

    gradys_config = GrADySEnvironmentConfig(**env_config)
    env: EnvBase = GradysTorchrlEnvironment(gradys_config)

    if not is_sequential:
        from torchrl.envs.transforms import CatTensors, FlattenObservation

        env = env.append_transform(
            FlattenObservation(
                first_dim=-2,
                last_dim=-1,
                in_keys=[
                    ("agents", "observation", "sensors"),
                    ("agents", "observation", "drones"),
                ],
                out_keys=[
                    ("agents", "observation_flat", "sensors"),
                    ("agents", "observation_flat", "drones"),
                ],
            )
        )
        include_id = env_config.get("id_on_state", True)
        in_keys = [
            ("agents", "observation_flat", "sensors"),
            ("agents", "observation_flat", "drones"),
        ]
        if include_id:
            in_keys.append(("agents", "observation", "agent_id"))
        env = env.append_transform(
            CatTensors(
                in_keys=in_keys,
                out_key=("agents", "observation"),
                del_keys=False,
            )
        )

    return env
