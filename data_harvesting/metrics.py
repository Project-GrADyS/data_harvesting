import time
import mlflow
import torch
from tensordict import TensorDictBase

from data_harvesting.environment import EndCause

class EnvironmentMetricsCollector:
    def __init__(self):
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

    def report_metrics(self, batch: TensorDictBase):
        # Compute mask of episodes that terminated at the last step (batch dimension)
        done_last = batch.get(("next", "agents", "done"))[:, -1]
        mask = done_last.reshape(-1).to(torch.bool)

        # Fast path: nothing to accumulate this call
        n_done = int(mask.sum().item())
        if n_done == 0:
            return

        self.trajectories += n_done

        info = batch.get(("next", "agents", "info"))[mask, 0]
        metric_sums = info.sum().detach().cpu()

        self.sum_avg_reward += metric_sums["avg_reward"].item()
        self.sum_max_reward += metric_sums["max_reward"].item()
        self.sum_sum_reward += metric_sums["sum_reward"].item()
        self.sum_avg_collection_time += metric_sums["avg_collection_time"].item()
        self.sum_episode_duration += metric_sums["episode_duration"].item()
        self.sum_completion_time += metric_sums["completion_time"].item()
        self.sum_all_collected += metric_sums["all_collected"].item()
        self.sum_num_collected += metric_sums["num_collected"].item()

    def log_metrics(self, step: int):        
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
        # Batch all metrics in a single call for performance
        metrics = {
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "sum_reward": sum_reward,
            "avg_collection_time": avg_collection_time,
            "episode_duration": episode_duration,
            "completion_time": completion_time,
            "all_collected": all_collected,
            "num_collected": num_collected,
        }
        # Include end-cause counters
        for cause, count in self.end_cause_counts.items():
            metrics[f"end_cause_{cause.name}"] = count

        mlflow.log_metrics(metrics, step=step)

class LearningMetricsCollector:
    def __init__(self):
        self.losses: dict[str, float] = {}
        self.iterations = 0
        self.start_time: float | None = None

    def report_loss(self, loss_name: str, loss_value: float):
        if loss_name not in self.losses:
            self.losses[loss_name] = 0
        self.losses[loss_name] += loss_value

        self.start_time = time.time()

        self.iterations += 1

    def log_metrics(self, step: int):
        # Batch all learning metrics in a single call
        metrics: dict[str, float] = {}
        for loss_name, loss_value in self.losses.items():
            avg_loss = loss_value / self.iterations
            metrics[f"loss_{loss_name}"] = avg_loss

        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            sps = 1 / elapsed_time if elapsed_time > 0 else 0
            metrics["sps"] = sps

        if metrics:
            mlflow.log_metrics(metrics, step=step)
        self.losses.clear()