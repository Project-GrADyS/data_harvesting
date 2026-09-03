from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest
import torch

import rl_core
import rl_core.collection as collection
from rl_core.collection import (
    CollectionMode,
    CollectorConfig,
    make_collector,
    validate_collector_config,
)


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

    monkeypatch.setattr(collection, "Collector", factory("sync-single"))
    monkeypatch.setattr(collection, "AsyncCollector", factory("async-single"))
    monkeypatch.setattr(collection, "MultiSyncCollector", factory("sync-multi"))
    monkeypatch.setattr(collection, "MultiAsyncCollector", factory("async-multi"))
    return created


def valid_config(**overrides: object) -> CollectorConfig:
    values = {
        "mode": CollectionMode.SYNC,
        "frames_per_batch": 8,
        "total_frames": 32,
        "num_workers": 1,
        "device": None,
        "env_device": None,
        "policy_device": None,
        "storing_device": None,
        "collector_kwargs": {},
    }
    values.update(overrides)
    return CollectorConfig(**values)  # type: ignore[arg-type]


def test_collector_config_is_passive_frozen_slotted_and_keyword_only() -> None:
    invalid = CollectorConfig(mode="sync", frames_per_batch=0)  # type: ignore[arg-type]

    assert invalid.mode == "sync"
    assert invalid.frames_per_batch == 0
    assert not hasattr(invalid, "__dict__")
    with pytest.raises(FrozenInstanceError):
        invalid.frames_per_batch = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        CollectorConfig(CollectionMode.SYNC, 8)  # type: ignore[misc]


def test_collector_config_keeps_caller_owned_kwargs_mapping() -> None:
    kwargs = {"set_truncated": True}
    config = valid_config(collector_kwargs=kwargs)

    assert config.collector_kwargs is kwargs

    kwargs["set_truncated"] = False
    assert config.collector_kwargs["set_truncated"] is False


def test_validate_collector_config_accepts_all_supported_boundaries() -> None:
    validate_collector_config(
        valid_config(
            mode=CollectionMode.ASYNC,
            frames_per_batch=1,
            total_frames=-1,
            num_workers=1,
            device=torch.device("cpu"),
            env_device="cpu",
            policy_device=None,
            storing_device="cuda:0",
            collector_kwargs={"set_truncated": True},
        )
    )
    validate_collector_config(valid_config(total_frames=1))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "sync"),
        ("frames_per_batch", 1.0),
        ("frames_per_batch", True),
        ("total_frames", 1.0),
        ("total_frames", False),
        ("num_workers", 1.0),
        ("num_workers", True),
        ("device", 0),
        ("env_device", object()),
        ("policy_device", False),
        ("storing_device", []),
        ("collector_kwargs", []),
    ],
)
def test_validate_collector_config_rejects_wrong_field_types(
    field: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match=field):
        validate_collector_config(valid_config(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frames_per_batch", 0),
        ("frames_per_batch", -1),
        ("total_frames", 0),
        ("total_frames", -2),
        ("num_workers", 0),
        ("num_workers", -1),
        ("device", ""),
        ("env_device", ""),
        ("policy_device", ""),
        ("storing_device", ""),
    ],
)
def test_validate_collector_config_rejects_invalid_field_values(
    field: str,
    value: int | str,
) -> None:
    with pytest.raises(ValueError, match=field):
        validate_collector_config(valid_config(**{field: value}))


@pytest.mark.parametrize(
    "reserved_name",
    [
        "create_env_fn",
        "device",
        "env_device",
        "frames_per_batch",
        "num_workers",
        "policy",
        "policy_device",
        "storing_device",
        "total_frames",
    ],
)
def test_validate_collector_config_rejects_every_wrapper_owned_kwarg(
    reserved_name: str,
) -> None:
    with pytest.raises(ValueError, match="wrapper-owned"):
        validate_collector_config(
            valid_config(collector_kwargs={reserved_name: object()})
        )


def test_validate_collector_config_requires_a_collector_config() -> None:
    with pytest.raises(TypeError, match="config"):
        validate_collector_config(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "num_workers", "expected_kind"),
    [
        (CollectionMode.SYNC, 1, "sync-single"),
        (CollectionMode.ASYNC, 1, "async-single"),
        (CollectionMode.SYNC, 3, "sync-multi"),
        (CollectionMode.ASYNC, 3, "async-multi"),
    ],
)
def test_make_collector_selects_mode_and_worker_count_and_forwards_arguments(
    monkeypatch: pytest.MonkeyPatch,
    mode: CollectionMode,
    num_workers: int,
    expected_kind: str,
) -> None:
    created = install_fake_collectors(monkeypatch)
    env_factory = lambda: None
    policy = object()
    kwargs = {"set_truncated": True}
    config = valid_config(
        mode=mode,
        num_workers=num_workers,
        device="cuda:0",
        env_device="cpu",
        policy_device="cuda:0",
        storing_device="cpu",
        collector_kwargs=kwargs,
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


def test_make_collector_validates_config_before_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = install_fake_collectors(monkeypatch)
    config = valid_config(frames_per_batch=0)

    with pytest.raises(ValueError, match="frames_per_batch"):
        with make_collector(config=config, env_factory=lambda: None):
            pass

    assert created == []


def test_make_collector_calls_public_config_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_collectors(monkeypatch)
    config = valid_config()
    validated: list[CollectorConfig] = []

    def spy(candidate: CollectorConfig) -> None:
        validated.append(candidate)

    monkeypatch.setattr(collection, "validate_collector_config", spy)

    with make_collector(config=config, env_factory=lambda: None):
        pass

    assert validated == [config]


def test_make_collector_shuts_down_once_when_body_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = install_fake_collectors(monkeypatch)

    with make_collector(config=valid_config(), env_factory=lambda: None):
        pass

    assert created[0].shutdown_calls == 1


def test_make_collector_shuts_down_once_when_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = install_fake_collectors(monkeypatch)

    with pytest.raises(RuntimeError, match="training failed"):
        with make_collector(config=valid_config(), env_factory=lambda: None):
            raise RuntimeError("training failed")

    assert created[0].shutdown_calls == 1


def test_make_collector_requires_callable_environment_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = install_fake_collectors(monkeypatch)

    with pytest.raises(TypeError, match="env_factory"):
        with make_collector(config=valid_config(), env_factory=None):  # type: ignore[arg-type]
            pass

    assert created == []


def test_collection_api_is_exported_from_package_root() -> None:
    assert rl_core.CollectionMode is CollectionMode
    assert rl_core.CollectorConfig is CollectorConfig
    assert rl_core.make_collector is make_collector
    assert rl_core.validate_collector_config is validate_collector_config
