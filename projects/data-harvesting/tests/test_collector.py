from contextlib import contextmanager

import pytest
import torch
from rl_core import CollectionMode
from torch import nn

from data_harvesting.collector import create_collector, make_collector_config


def _config(*, async_collector: bool, num_collectors: int) -> dict:
    return {
        "collector": {
            "async_collector": async_collector,
            "num_collectors": num_collectors,
            "frames_per_batch": 32,
        },
        "training": {"total_timesteps": 256},
    }


@pytest.mark.parametrize(
    "async_collector,expected_mode",
    [(False, CollectionMode.SYNC), (True, CollectionMode.ASYNC)],
)
def test_project_translates_collector_configuration_to_rl_core(
    async_collector: bool, expected_mode: CollectionMode
) -> None:
    translated = make_collector_config(
        _config(async_collector=async_collector, num_collectors=3),
        torch.device("cpu"),
    )

    assert translated.mode is expected_mode
    assert translated.num_workers == 3
    assert translated.frames_per_batch == 32
    assert translated.total_frames == 256
    assert translated.device == torch.device("cpu")
    assert translated.env_device == "cpu"
    assert translated.policy_device == torch.device("cpu")


def test_project_delegates_collector_lifecycle_to_rl_core(monkeypatch) -> None:
    policy = nn.Identity()
    environment_factory = lambda: object()
    collector = object()
    captured = {}

    @contextmanager
    def _fake_make_collector(*, config, env_factory, policy):
        captured.update(config=config, env_factory=env_factory, policy=policy)
        yield collector

    monkeypatch.setattr("data_harvesting.collector.make_collector", _fake_make_collector)

    with create_collector(
        policy,
        "cpu",
        environment_factory,
        _config(async_collector=False, num_collectors=1),
    ) as result:
        assert result is collector

    assert captured["config"].mode is CollectionMode.SYNC
    assert captured["env_factory"] is environment_factory
    assert captured["policy"] is policy
