from __future__ import annotations

from collections.abc import Callable

import pytest

import rl_core
import rl_core.collection as collection
from rl_core.collection import CollectionMode, CollectorConfig, make_collector


class FakeCollector:
    def __init__(self, kind: str, kwargs: dict) -> None:
        self.kind = kind
        self.kwargs = kwargs
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def install_fake_collectors(monkeypatch: pytest.MonkeyPatch) -> list[FakeCollector]:
    created: list[FakeCollector] = []

    def factory(kind: str) -> Callable[..., FakeCollector]:
        def build(**kwargs) -> FakeCollector:
            collector = FakeCollector(kind, kwargs)
            created.append(collector)
            return collector

        return build

    monkeypatch.setattr(collection, "SyncDataCollector", factory("sync-single"))
    monkeypatch.setattr(collection, "aSyncDataCollector", factory("async-single"))
    monkeypatch.setattr(collection, "MultiSyncDataCollector", factory("sync-multi"))
    monkeypatch.setattr(collection, "MultiaSyncDataCollector", factory("async-multi"))
    return created


@pytest.mark.parametrize(
    ("mode", "num_workers", "expected_kind"),
    [
        (CollectionMode.SYNC, 1, "sync-single"),
        (CollectionMode.ASYNC, 1, "async-single"),
        (CollectionMode.SYNC, 3, "sync-multi"),
        (CollectionMode.ASYNC, 3, "async-multi"),
    ],
)
def test_make_collector_selects_mode_and_worker_count(
    monkeypatch: pytest.MonkeyPatch,
    mode: CollectionMode,
    num_workers: int,
    expected_kind: str,
) -> None:
    created = install_fake_collectors(monkeypatch)
    env_factory = lambda: None
    policy = object()
    config = CollectorConfig(
        mode=mode,
        frames_per_batch=8,
        total_frames=32,
        num_workers=num_workers,
        device="cuda:0",
        env_device="cpu",
        policy_device="cuda:0",
        storing_device="cpu",
        collector_kwargs={"set_truncated": True},
    )

    with make_collector(config=config, env_factory=env_factory, policy=policy) as collector:
        assert collector.kind == expected_kind
        assert collector.shutdown_calls == 0

    assert len(created) == 1
    built = created[0]
    assert built.shutdown_calls == 1
    assert built.kwargs["policy"] is policy
    assert built.kwargs["frames_per_batch"] == 8
    assert built.kwargs["total_frames"] == 32
    assert built.kwargs["device"] == "cuda:0"
    assert built.kwargs["env_device"] == "cpu"
    assert built.kwargs["policy_device"] == "cuda:0"
    assert built.kwargs["storing_device"] == "cpu"
    assert built.kwargs["set_truncated"] is True
    if num_workers == 1:
        assert built.kwargs["create_env_fn"] is env_factory
    else:
        assert built.kwargs["create_env_fn"] == [env_factory] * num_workers


def test_make_collector_shuts_down_when_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_fake_collectors(monkeypatch)
    config = CollectorConfig(mode=CollectionMode.SYNC, frames_per_batch=4, total_frames=8)

    with pytest.raises(RuntimeError, match="training failed"):
        with make_collector(config=config, env_factory=lambda: None):
            raise RuntimeError("training failed")

    assert created[0].shutdown_calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "sync", "frames_per_batch": 4},
        {"mode": CollectionMode.SYNC, "frames_per_batch": 0},
        {"mode": CollectionMode.SYNC, "frames_per_batch": 4, "num_workers": 0},
        {"mode": CollectionMode.SYNC, "frames_per_batch": 4, "total_frames": 0},
        {
            "mode": CollectionMode.SYNC,
            "frames_per_batch": 4,
            "collector_kwargs": {"total_frames": 20},
        },
    ],
)
def test_collector_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        CollectorConfig(**kwargs)


def test_collector_config_accepts_endless_collection() -> None:
    config = CollectorConfig(mode=CollectionMode.ASYNC, frames_per_batch=4, total_frames=-1)

    assert config.total_frames == -1


def test_collector_config_allows_partial_final_batch() -> None:
    config = CollectorConfig(mode=CollectionMode.SYNC, frames_per_batch=4, total_frames=10)

    assert config.total_frames == 10


def test_collector_kwargs_are_defensively_copied() -> None:
    kwargs = {"set_truncated": True}
    config = CollectorConfig(
        mode=CollectionMode.SYNC,
        frames_per_batch=4,
        collector_kwargs=kwargs,
    )

    kwargs["set_truncated"] = False

    assert config.collector_kwargs["set_truncated"] is True


def test_make_collector_requires_callable_environment_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_collectors(monkeypatch)
    config = CollectorConfig(mode=CollectionMode.SYNC, frames_per_batch=4)

    with pytest.raises(TypeError, match="env_factory"):
        with make_collector(config=config, env_factory=None):  # type: ignore[arg-type]
            pass


def test_collection_api_is_exported_from_package_root() -> None:
    assert rl_core.CollectionMode is CollectionMode
    assert rl_core.CollectorConfig is CollectorConfig
    assert rl_core.make_collector is make_collector
