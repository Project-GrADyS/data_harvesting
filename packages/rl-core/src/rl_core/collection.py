from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeAlias

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.collectors import (
    DataCollectorBase,
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
    SyncDataCollector,
    aSyncDataCollector,
)
from torchrl.envs import EnvBase
from validation_core import (
    validate_callable,
    validate_mapping,
    validate_non_empty_string,
    validate_positive_integer,
)


class CollectionMode(StrEnum):
    """Whether workers pause while the caller consumes each collected batch."""

    SYNC = "sync"
    ASYNC = "async"


Policy: TypeAlias = nn.Module | Callable[[TensorDictBase], TensorDictBase] | None
EnvironmentFactory: TypeAlias = Callable[[], EnvBase]


_RESERVED_COLLECTOR_KWARGS = {
    "create_env_fn",
    "device",
    "env_device",
    "frames_per_batch",
    "num_workers",
    "policy",
    "policy_device",
    "storing_device",
    "total_frames",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectorConfig:
    """Configuration shared by TorchRL's single- and multi-worker collectors."""

    mode: CollectionMode
    frames_per_batch: int
    total_frames: int = -1
    num_workers: int = 1
    device: torch.device | str | None = None
    env_device: torch.device | str | None = None
    policy_device: torch.device | str | None = None
    storing_device: torch.device | str | None = None
    collector_kwargs: Mapping[str, Any] = field(default_factory=dict)



def validate_collector_config(config: CollectorConfig) -> None:
    """Validate a collector configuration before constructing a collector."""

    if not isinstance(config, CollectorConfig):
        raise TypeError(f"config must be a CollectorConfig, got {type(config)}.")
    if not isinstance(config.mode, CollectionMode):
        raise TypeError(f"mode must be a CollectionMode, got {type(config.mode)}.")
    validate_positive_integer("frames_per_batch", config.frames_per_batch)
    validate_positive_integer("num_workers", config.num_workers)
    if config.total_frames != -1:
        validate_positive_integer("total_frames", config.total_frames)
    for field_name in ("device", "env_device", "policy_device", "storing_device"):
        device = getattr(config, field_name)
        if device is None or isinstance(device, torch.device):
            continue
        if isinstance(device, str):
            validate_non_empty_string(field_name, device)
            continue
        raise TypeError(f"{field_name} must be a torch.device, string, or None.")
    validate_mapping("collector_kwargs", config.collector_kwargs)
    conflicts = _RESERVED_COLLECTOR_KWARGS.intersection(config.collector_kwargs)
    if conflicts:
        raise ValueError(f"collector_kwargs contains wrapper-owned arguments: {sorted(conflicts)}")


def _build_collector(
    *,
    config: CollectorConfig,
    env_factory: EnvironmentFactory,
    policy: Policy,
) -> DataCollectorBase:
    common_kwargs = {
        "policy": policy,
        "frames_per_batch": config.frames_per_batch,
        "total_frames": config.total_frames,
        "device": config.device,
        "env_device": config.env_device,
        "policy_device": config.policy_device,
        "storing_device": config.storing_device,
        **config.collector_kwargs,
    }

    if config.num_workers == 1:
        collector_type = aSyncDataCollector if config.mode is CollectionMode.ASYNC else SyncDataCollector
        return collector_type(create_env_fn=env_factory, **common_kwargs)

    collector_type = MultiaSyncDataCollector if config.mode is CollectionMode.ASYNC else MultiSyncDataCollector
    env_factories = [env_factory] * config.num_workers
    return collector_type(create_env_fn=env_factories, **common_kwargs)


@contextmanager
def make_collector(
    *,
    config: CollectorConfig,
    env_factory: EnvironmentFactory,
    policy: Policy = None,
) -> Iterator[DataCollectorBase]:
    """Construct and reliably shut down the configured TorchRL collector.

    Multi-worker collectors must be created under Python's ``if __name__ == "__main__"`` guard.
    The caller owns iteration, batch processing, and policy-weight update timing.
    """

    validate_collector_config(config)
    validate_callable("env_factory", env_factory)
    collector = _build_collector(config=config, env_factory=env_factory, policy=policy)
    try:
        yield collector
    finally:
        collector.shutdown()


__all__ = ["CollectionMode", "CollectorConfig", "make_collector", "validate_collector_config"]
