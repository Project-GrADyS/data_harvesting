from __future__ import annotations

from collections import UserDict

import pytest

import rl_core
import rl_core.scheduling as scheduling
from rl_core.scheduling import ScheduledCallback, Scheduler


def test_scheduler_starts_at_zero_and_default_step_needs_no_callbacks() -> None:
    scheduler = Scheduler()

    scheduler.step()

    assert scheduler.current_step == 1


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


@pytest.mark.parametrize("name", [None, 1, True])
def test_unregister_rejects_non_string_name(name: object) -> None:
    scheduler = Scheduler()

    with pytest.raises(TypeError, match="name must be a string"):
        scheduler.unregister(name)  # type: ignore[arg-type]


def test_unregister_rejects_empty_name() -> None:
    scheduler = Scheduler()

    with pytest.raises(ValueError, match="name.*empty"):
        scheduler.unregister("")


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


def test_state_dict_returns_a_fresh_plain_dict() -> None:
    scheduler = Scheduler()
    scheduler.step(increment=3)

    first = scheduler.state_dict()
    second = scheduler.state_dict()
    first["current_step"] = 99

    assert type(second) is dict
    assert second == {"current_step": 3}
    assert scheduler.current_step == 3


@pytest.mark.parametrize("every", [True, 1.5, "1", None])
def test_register_rejects_non_integer_frequency(every: object) -> None:
    scheduler = Scheduler()

    with pytest.raises(TypeError, match="every must be an integer"):
        scheduler.register("invalid", every=every, callback=lambda step: None)  # type: ignore[arg-type]


@pytest.mark.parametrize("every", [0, -1])
def test_register_rejects_non_positive_frequency(every: int) -> None:
    scheduler = Scheduler()

    with pytest.raises(ValueError, match="every must be positive"):
        scheduler.register("invalid", every=every, callback=lambda step: None)


@pytest.mark.parametrize("name", [None, 1, True])
def test_register_rejects_non_string_name(name: object) -> None:
    scheduler = Scheduler()

    with pytest.raises(TypeError, match="name must be a string"):
        scheduler.register(name, every=1, callback=lambda step: None)  # type: ignore[arg-type]


def test_register_rejects_empty_name() -> None:
    scheduler = Scheduler()

    with pytest.raises(ValueError, match="name.*empty"):
        scheduler.register("", every=1, callback=lambda step: None)


def test_register_validates_callback_and_uniqueness() -> None:
    scheduler = Scheduler()

    with pytest.raises(TypeError, match="callable"):
        scheduler.register("invalid", every=1, callback=None)  # type: ignore[arg-type]

    scheduler.register("metrics", every=1, callback=lambda step: None)
    with pytest.raises(ValueError, match="already registered"):
        scheduler.register("metrics", every=2, callback=lambda step: None)


@pytest.mark.parametrize("increment", [True, 1.5, "1", None])
def test_step_rejects_non_integer_increment(increment: object) -> None:
    scheduler = Scheduler()

    with pytest.raises(TypeError, match="increment must be an integer"):
        scheduler.step(increment)  # type: ignore[arg-type]
    assert scheduler.current_step == 0


@pytest.mark.parametrize("increment", [0, -1])
def test_step_rejects_non_positive_increment(increment: int) -> None:
    scheduler = Scheduler()

    with pytest.raises(ValueError, match="increment must be positive"):
        scheduler.step(increment)
    assert scheduler.current_step == 0


@pytest.mark.parametrize("state", [None, 1, [], "current_step"])
def test_load_state_dict_rejects_non_mapping(state: object) -> None:
    scheduler = Scheduler()

    with pytest.raises(TypeError, match="state_dict must be a mapping"):
        scheduler.load_state_dict(state)  # type: ignore[arg-type]
    assert scheduler.current_step == 0


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({}, "missing keys"),
        ({"current_step": 1, "extra": 2}, "unexpected keys"),
        ({"extra": 2}, "missing keys"),
    ],
)
def test_load_state_dict_requires_exact_keys(
    state: dict[str, object], message: str
) -> None:
    scheduler = Scheduler()

    with pytest.raises(ValueError, match=message):
        scheduler.load_state_dict(state)
    assert scheduler.current_step == 0


@pytest.mark.parametrize("current_step", [True, 1.5, "1", None])
def test_load_state_dict_rejects_non_integer_current_step(current_step: object) -> None:
    scheduler = Scheduler()

    with pytest.raises(TypeError, match="current_step must be an integer"):
        scheduler.load_state_dict({"current_step": current_step})
    assert scheduler.current_step == 0


def test_load_state_dict_rejects_negative_current_step() -> None:
    scheduler = Scheduler()

    with pytest.raises(ValueError, match="current_step must be non-negative"):
        scheduler.load_state_dict({"current_step": -1})
    assert scheduler.current_step == 0


@pytest.mark.parametrize(
    ("state", "expected_step"),
    [
        ({"current_step": 0}, 0),
        (UserDict({"current_step": 4}), 4),
    ],
)
def test_load_state_dict_accepts_non_negative_step_from_any_mapping(
    state: dict[str, int] | UserDict[str, int], expected_step: int
) -> None:
    scheduler = Scheduler()

    scheduler.load_state_dict(state)

    assert scheduler.current_step == expected_step


@pytest.mark.parametrize(
    "invalid_state",
    [
        {},
        {"current_step": 1, "extra": 2},
        {"current_step": True},
        {"current_step": -1},
    ],
)
def test_failed_load_state_dict_leaves_clock_unchanged(
    invalid_state: dict[str, object],
) -> None:
    scheduler = Scheduler()
    scheduler.step(increment=7)

    with pytest.raises((TypeError, ValueError)):
        scheduler.load_state_dict(invalid_state)

    assert scheduler.current_step == 7


def test_scheduling_public_exports_are_exact() -> None:
    assert scheduling.__all__ == ["ScheduledCallback", "Scheduler"]


def test_scheduler_types_are_exported_from_package_root() -> None:
    assert rl_core.Scheduler is Scheduler
    assert rl_core.ScheduledCallback is ScheduledCallback
