from __future__ import annotations

import pytest

from rl_core.checkpointing import Checkpoint, CheckpointManager


class RecordingStore:
    def __init__(self, name: str, calls: list[str], error: Exception | None = None) -> None:
        self.name = name
        self.calls = calls
        self.error = error

    def save(self, checkpoint: Checkpoint) -> str:
        self.calls.append(self.name)
        if self.error is not None:
            raise self.error
        return f"{self.name}:{checkpoint.step}"


def test_manager_requires_an_iterable_of_stores() -> None:
    with pytest.raises(TypeError):
        CheckpointManager(None)  # type: ignore[arg-type]


def test_manager_requires_at_least_one_store() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CheckpointManager([])


@pytest.mark.parametrize("store", [object(), None, "store"])
def test_manager_rejects_stores_without_callable_save(store: object) -> None:
    with pytest.raises(TypeError, match="save"):
        CheckpointManager([store])  # type: ignore[list-item]


def test_manager_consumes_a_one_shot_store_iterable_once() -> None:
    calls: list[str] = []
    stores = (RecordingStore(name, calls) for name in ("local", "remote"))

    manager = CheckpointManager(stores)
    results = manager.save(Checkpoint(step=7, state={"policy": {}}))

    assert results == ("local:7", "remote:7")
    assert calls == ["local", "remote"]


def test_manager_saves_to_stores_in_order(checkpoint_factory) -> None:
    calls: list[str] = []
    manager = CheckpointManager(
        [RecordingStore("local", calls), RecordingStore("remote", calls)]
    )

    assert manager.save(checkpoint_factory(step=7)) == ("local:7", "remote:7")
    assert calls == ["local", "remote"]


def test_manager_stops_at_and_propagates_store_failure(checkpoint_factory) -> None:
    calls: list[str] = []
    manager = CheckpointManager(
        [
            RecordingStore("first", calls),
            RecordingStore("failing", calls, RuntimeError("store unavailable")),
            RecordingStore("later", calls),
        ]
    )

    with pytest.raises(RuntimeError, match="store unavailable"):
        manager.save(checkpoint_factory(step=1))

    assert calls == ["first", "failing"]


def test_manager_rejects_non_checkpoint_input() -> None:
    manager = CheckpointManager([RecordingStore("local", [])])

    with pytest.raises(TypeError, match="checkpoint"):
        manager.save({"step": 1})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("checkpoint", "exception"),
    [
        (Checkpoint(step="1", state={"policy": {}}), TypeError),  # type: ignore[arg-type]
        (Checkpoint(step=-1, state={"policy": {}}), ValueError),
        (Checkpoint(step=1, state={}), ValueError),
        (Checkpoint(step=1, state={"policy": {}}, metadata=[]), TypeError),  # type: ignore[arg-type]
    ],
)
def test_manager_validates_checkpoint_before_calling_any_store(
    checkpoint: Checkpoint, exception: type[Exception]
) -> None:
    calls: list[str] = []
    manager = CheckpointManager([RecordingStore("local", calls)])

    with pytest.raises(exception):
        manager.save(checkpoint)

    assert calls == []
