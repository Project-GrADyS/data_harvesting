import time
import torch
from dvclive import Live
from tensordict import TensorDictBase

from data_harvesting.environment import EndCause

class LiveSwitch:
    def __init__(self, enabled: bool = True, *args, **kwargs):
        self.live = Live(*args, **kwargs) if enabled else None

    def __enter__(self):
        if self.live:
            self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.live:
            self.live.__exit__(exc_type, exc_value, traceback)

    def __getattr__(self, name):
        if self.live:
            return getattr(self.live, name)
        else:
            def noop(*args, **kwargs):
                pass
            return noop

    

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
        # Compute mask of episodes that terminated at the last step (batch dimension)
        done_last = batch.get(("next", "agents", "done"))[:, -1]
        mask = done_last.reshape(-1).to(torch.bool)

        # Fast path: nothing to accumulate this call
        n_done = int(mask.sum().item())
        if n_done == 0:
            return

        self.trajectories += n_done

        info = batch.get(("next", "agents", "info"))[mask, 0]
        metric_sums = info.sum().cpu()

        self.sum_avg_reward += metric_sums["avg_reward"].item()
        self.sum_max_reward += metric_sums["max_reward"].item()
        self.sum_sum_reward += metric_sums["sum_reward"].item()
        self.sum_avg_collection_time += metric_sums["avg_collection_time"].item()
        self.sum_episode_duration += metric_sums["episode_duration"].item()
        self.sum_completion_time += metric_sums["completion_time"].item()
        self.sum_all_collected += metric_sums["all_collected"].item()
        self.sum_num_collected += metric_sums["num_collected"].item()

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