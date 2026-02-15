import time
import mlflow
import torch
from tensordict import TensorDictBase

from data_harvesting.environment import EndCause

class EnvironmentMetricsCollector:
    def __init__(self, device: torch.device):
        self._device = device
        self.trajectories: torch.Tensor = torch.zeros((), device=device)

        self.sum_avg_reward: torch.Tensor = torch.zeros((), device=device)
        self.sum_max_reward: torch.Tensor = torch.zeros((), device=device)
        self.sum_sum_reward: torch.Tensor = torch.zeros((), device=device)
        self.sum_avg_collection_time: torch.Tensor = torch.zeros((), device=device)
        self.sum_episode_duration: torch.Tensor = torch.zeros((), device=device)
        self.sum_completion_time: torch.Tensor = torch.zeros((), device=device)
        self.sum_all_collected: torch.Tensor = torch.zeros((), device=device)
        self.sum_num_collected: torch.Tensor = torch.zeros((), device=device)
        self.end_cause_counts: torch.Tensor = torch.zeros(len(EndCause), device=device)

    def report_metrics(self, batch: TensorDictBase):
        # Compute mask of episodes that terminated at the last step (batch dimension)
        done_last = batch.get(("next", "agents", "done"))[:, -1]
        mask = done_last.reshape(-1).to(torch.bool)

        info = batch.get(("next", "agents", "info"))[mask, 0]

        # Accumulate sums on-device to avoid per-step syncs.
        det_info = info.detach()
        metric_sums = det_info.sum(dim=0)
        self.trajectories += mask.sum()

        self.sum_avg_reward += metric_sums["avg_reward"]
        self.sum_max_reward += metric_sums["max_reward"]
        self.sum_sum_reward += metric_sums["sum_reward"]
        self.sum_avg_collection_time += metric_sums["avg_collection_time"]
        self.sum_episode_duration += metric_sums["episode_duration"]
        self.sum_completion_time += metric_sums["completion_time"]
        self.sum_all_collected += metric_sums["all_collected"]
        self.sum_num_collected += metric_sums["num_collected"]
        # Update end-cause counts
        end_cause = info["end_cause"]
        for i, cause in enumerate(EndCause):
            self.end_cause_counts[i] += (end_cause == cause.value).sum()

    def log_metrics(self, step: int):        
        trajectories = self.trajectories.item()
        if trajectories == 0:
            return

        avg_reward = (self.sum_avg_reward / self.trajectories).item()
        max_reward = (self.sum_max_reward / self.trajectories).item()
        sum_reward = (self.sum_sum_reward / self.trajectories).item()
        avg_collection_time = (self.sum_avg_collection_time / self.trajectories).item()
        episode_duration = (self.sum_episode_duration / self.trajectories).item()
        completion_time = (self.sum_completion_time / self.trajectories).item()
        all_collected = (self.sum_all_collected / self.trajectories).item()
        num_collected = (self.sum_num_collected / self.trajectories).item()
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
        for i, count in enumerate(self.end_cause_counts):
            metrics[f"end_cause_{EndCause(i).name}"] = count.item()

        mlflow.log_metrics(metrics, step=step)

class LearningMetricsCollector:
    def __init__(self, device: torch.device):
        self._device = device
        self.losses: dict[str, torch.Tensor] = {}
        self.iterations: torch.Tensor = torch.zeros((), device=device)
        self.start_time: float | None = None

    def report_loss(self, loss_name: str, loss_value: torch.Tensor):
        if loss_name not in self.losses:
            self.losses[loss_name] = torch.zeros((), device=self._device)
        self.losses[loss_name] += loss_value.detach()

        self.start_time = time.time()

        self.iterations += 1

    def log_metrics(self, step: int):
        # Batch all learning metrics in a single call
        metrics: dict[str, float] = {}
        iterations = self.iterations.item()
        if iterations == 0:
            return

        for loss_name, loss_value in self.losses.items():
            avg_loss = (loss_value / self.iterations).item()
            metrics[f"loss_{loss_name}"] = avg_loss

        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            sps = 1 / elapsed_time if elapsed_time > 0 else 0
            metrics["sps"] = sps

        if metrics:
            mlflow.log_metrics(metrics, step=step)
        self.losses.clear()
        self.iterations.zero_()