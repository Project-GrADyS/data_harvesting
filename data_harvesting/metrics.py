import time
from dvclive import Live
from tensordict import TensorDictBase

from data_harvesting.environment import EndCause

class EnvironmentMetricsCollector:
    def __init__(self, live: Live):
        self.live = live
        self.trajectories = 0

        self.sum_avg_reward = 0.0
        self.sum_max_reward = 0.0
        self.sum_sum_reward = 0.0
        self.sum_avg_collection_time = 0.0
        self.sum_episode_duration = 0.0
        self.sum_completion_time = 0.0
        self.sum_all_collected = 0.0
        self.sum_num_collected = 0.0
        self.end_cause_counts = {
            cause: 0 for cause in EndCause
        }

    def _accumulate_metrics(self, batch: TensorDictBase):
        done_steps = batch[batch.get(("next", "agents", "done"))[:, -1].reshape(-1)]

        self.trajectories += done_steps.shape[0]

        self.sum_avg_reward += done_steps.get(("next", "agents", "info", "avg_reward"))[:, -1].sum().item()
        self.sum_max_reward += done_steps.get(("next", "agents", "info", "max_reward"))[:, -1].sum().item()
        self.sum_sum_reward += done_steps.get(("next", "agents", "info", "sum_reward"))[:, -1].sum().item()
        self.sum_avg_collection_time += done_steps.get(("next", "agents", "info", "avg_collection_time"))[:, -1].sum().item()
        self.sum_episode_duration += done_steps.get(("next", "agents", "info", "episode_duration"))[:, -1].sum().item()
        self.sum_completion_time += done_steps.get(("next", "agents", "info", "completion_time"))[:, -1].sum().item()
        self.sum_all_collected += done_steps.get(("next", "agents", "info", "all_collected"))[:, -1].sum().item()
        self.sum_num_collected += done_steps.get(("next", "agents", "info", "num_collected"))[:, -1].sum().item()

        # Update end cause counts
        causes = done_steps.get(("next", "agents", "info", "cause"))[:, -1].flatten().tolist()
        for cause in causes:
            if cause in self.end_cause_counts:
                self.end_cause_counts[cause] += 1
            else:
                self.end_cause_counts[cause] = 1

    def log_metrics(self, batch: TensorDictBase):
        self._accumulate_metrics(batch)
        
        if self.trajectories == 0:
            return  # Avoid division by zero

        avg_reward = self.sum_avg_reward / self.trajectories
        max_reward = self.sum_max_reward / self.trajectories
        sum_reward = self.sum_sum_reward / self.trajectories
        avg_collection_time = self.sum_avg_collection_time / self.trajectories
        episode_duration = self.sum_episode_duration / self.trajectories
        completion_time = self.sum_completion_time / self.trajectories
        all_collected = self.sum_all_collected / self.trajectories
        num_collected = self.sum_num_collected / self.trajectories

        self.live.log_metric("avg_reward", avg_reward)
        self.live.log_metric("max_reward", max_reward)
        self.live.log_metric("sum_reward", sum_reward)
        self.live.log_metric("avg_collection_time", avg_collection_time)
        self.live.log_metric("episode_duration", episode_duration)
        self.live.log_metric("completion_time", completion_time)
        self.live.log_metric("all_collected", all_collected)
        self.live.log_metric("num_collected", num_collected)

        for cause, count in self.end_cause_counts.items():
            cause_enum = EndCause(cause)
            self.live.log_metric(f"end_cause_{cause_enum.name}", count)

class LearningMetricsCollector:
    def __init__(self, live: Live):
        self.live = live
        self.losses: dict[str, list[float]] = {}
        self.start_time: float | None = None

    def report_loss(self, loss_name: str, loss_value: float):
        if loss_name not in self.losses:
            self.losses[loss_name] = []
        self.losses[loss_name].append(loss_value)

        if self.start_time is None:
            self.start_time = time.time()

    def log_metrics(self):
        for loss_name, values in self.losses.items():
            if len(values) > 0:
                avg_loss = sum(values) / len(values)
                self.live.log_metric(f"loss_{loss_name}", avg_loss)

        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            sps = 1 / elapsed_time if elapsed_time > 0 else 0
            self.live.log_metric("sps", sps)
        self.losses.clear()