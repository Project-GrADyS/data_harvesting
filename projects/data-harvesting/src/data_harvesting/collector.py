from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from rl_core import CollectionMode, CollectorConfig, make_collector
from tensordict.nn import TensorDictModule
from torchrl.collectors import BaseCollector
from torchrl.envs import EnvBase


def make_collector_config(config: Mapping[str, Any], device: torch.device | str) -> CollectorConfig:
    collector_config = config["collector"]
    return CollectorConfig(
        mode=(
            CollectionMode.ASYNC
            if bool(collector_config["async_collector"])
            else CollectionMode.SYNC
        ),
        frames_per_batch=int(collector_config["frames_per_batch"]),
        total_frames=int(config["training"]["total_timesteps"]),
        num_workers=int(collector_config["num_collectors"]),
        device=device,
        env_device="cpu",
        policy_device=device,
    )


@contextmanager
def create_collector(
    exploratory_policy: TensorDictModule,
    device: torch.device | str,
    env_creator: Callable[[], EnvBase],
    config: Mapping[str, Any],
) -> Iterator[BaseCollector]:
    with make_collector(
        config=make_collector_config(config, device),
        env_factory=env_creator,
        policy=exploratory_policy,
    ) as collector:
        yield collector
