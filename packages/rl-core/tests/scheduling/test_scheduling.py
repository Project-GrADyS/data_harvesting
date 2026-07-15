from __future__ import annotations

import pytest

import rl_core
from rl_core.scheduling import Scheduler


def test_callbacks_run_at_their_frequencies() -> None:
    scheduler = Scheduler()
    calls: list[tuple[str, int]] = []
    scheduler.register("fast", every=2, callback=lambda step: calls.append(("fast", step)))
    scheduler.register("slow", every=3, callback=lambda step: calls.append(("slow", step)))

    for _ in range(6):
        scheduler.step()

    assert scheduler.current_step == 6
    assert calls == [
        ("fast", 2),
        ("slow", 3),
        ("fast", 4),
        ("fast", 6),
        ("slow", 6),
    ]


def test_callbacks_due_together_run_in_registration_order() -> None:
    scheduler = Scheduler()
    calls: list[str] = []
    scheduler.register("first", every=2, callback=lambda step: calls.append("first"))
    scheduler.register("second", every=2, callback=lambda step: calls.append("second"))

    scheduler.step(increment=2)

    assert calls == ["first", "second"]


def test_increment_crosses_intervals_and_invokes_each_callback_once() -> None:
    scheduler = Scheduler()
    calls: list[int] = []
    scheduler.register("checkpoint", every=10, callback=calls.append)

    scheduler.step(increment=25)
    scheduler.step(increment=4)
    scheduler.step()

    assert scheduler.current_step == 30
    assert calls == [25, 30]


def test_unregister_removes_callback() -> None:
    scheduler = Scheduler()
    calls: list[int] = []
    scheduler.register("metrics", every=1, callback=calls.append)

    scheduler.step()
    scheduler.unregister("metrics")
    scheduler.step()

    assert calls == [1]
    with pytest.raises(KeyError, match="No callback"):
        scheduler.unregister("metrics")


def test_registration_mutations_during_dispatch_apply_on_following_step() -> None:
    scheduler = Scheduler()
    calls: list[tuple[str, int]] = []

    def mutate_registrations(step: int) -> None:
        calls.append(("first", step))
        if step == 1:
            scheduler.unregister("second")
            scheduler.register("third", every=1, callback=lambda value: calls.append(("third", value)))

    scheduler.register("first", every=1, callback=mutate_registrations)
    scheduler.register("second", every=1, callback=lambda step: calls.append(("second", step)))

    scheduler.step()
    scheduler.step()

    assert calls == [("first", 1), ("second", 1), ("first", 2), ("third", 2)]


def test_callback_failure_stops_dispatch_without_rolling_back_or_retrying() -> None:
    scheduler = Scheduler()
    calls: list[tuple[str, int]] = []

    def fail(step: int) -> None:
        calls.append(("failing", step))
        raise RuntimeError("callback failed")

    scheduler.register("failing", every=2, callback=fail)
    scheduler.register("later", every=2, callback=lambda step: calls.append(("later", step)))

    with pytest.raises(RuntimeError, match="callback failed"):
        scheduler.step(increment=2)

    assert scheduler.current_step == 2
    assert calls == [("failing", 2)]

    scheduler.unregister("failing")
    scheduler.step()
    assert calls == [("failing", 2)]

    scheduler.step()
    assert calls == [("failing", 2), ("later", 4)]


def test_state_dict_restores_clock_but_not_registrations() -> None:
    scheduler = Scheduler()
    scheduler.step(increment=7)

    state = scheduler.state_dict()
    restored = Scheduler()
    calls: list[int] = []
    restored.register("evaluation", every=10, callback=calls.append)
    restored.load_state_dict(state)
    state["current_step"] = 999

    assert restored.current_step == 7
    restored.step(increment=3)
    assert calls == [10]


@pytest.mark.parametrize("every", [0, -1, True, 1.5])
def test_register_rejects_invalid_frequency(every: object) -> None:
    scheduler = Scheduler()

    with pytest.raises(ValueError, match="positive integer"):
        scheduler.register("invalid", every=every, callback=lambda step: None)  # type: ignore[arg-type]


def test_register_validates_name_callback_and_uniqueness() -> None:
    scheduler = Scheduler()

    with pytest.raises(ValueError, match="non-empty"):
        scheduler.register("", every=1, callback=lambda step: None)
    with pytest.raises(TypeError, match="callable"):
        scheduler.register("invalid", every=1, callback=None)  # type: ignore[arg-type]

    scheduler.register("metrics", every=1, callback=lambda step: None)
    with pytest.raises(ValueError, match="already registered"):
        scheduler.register("metrics", every=2, callback=lambda step: None)


@pytest.mark.parametrize("increment", [0, -1, True, 1.5])
def test_step_rejects_invalid_increment(increment: object) -> None:
    scheduler = Scheduler()

    with pytest.raises(ValueError, match="positive integer"):
        scheduler.step(increment)  # type: ignore[arg-type]
    assert scheduler.current_step == 0


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"current_step": 1, "extra": 2},
        {"current_step": -1},
        {"current_step": True},
    ],
)
def test_load_state_dict_rejects_invalid_state(state: dict[str, object]) -> None:
    scheduler = Scheduler()

    with pytest.raises(ValueError):
        scheduler.load_state_dict(state)
    assert scheduler.current_step == 0


def test_scheduler_is_exported_from_package_root() -> None:
    assert rl_core.Scheduler is Scheduler
