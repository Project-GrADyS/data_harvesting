from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from tensordict import TensorDict

from rl_core.metrics import (
    CategoricalMetricSpec,
    MetricsCollector,
    ScalarMetricSpec,
    ScalarReducer,
)


def make_collector(*, loggers=()) -> MetricsCollector:
    return MetricsCollector(
        specs=[
            ScalarMetricSpec(key="reward", reducer=ScalarReducer.MEAN),
            ScalarMetricSpec(key="items", reducer=ScalarReducer.SUM, output_name="items_total"),
            CategoricalMetricSpec(
                key="cause",
                output_prefix="end_cause",
                value_labels={1: "TIMEOUT", 2: "COMPLETED"},
            ),
        ],
        loggers=loggers,
    )


def test_push_and_peek_aggregate_independent_sample_counts() -> None:
    collector = make_collector()

    collector.push({"reward": torch.tensor([1.0, 2.0, 3.0]), "items": [2, 4]})
    collector.push({"reward": 6.0, "cause": torch.tensor([2, 1, 2])})

    assert collector.peek() == {
        "reward": pytest.approx(3.0),
        "items_total": pytest.approx(6.0),
        "end_cause_TIMEOUT": pytest.approx(1.0),
        "end_cause_COMPLETED": pytest.approx(2.0),
    }


def test_peek_does_not_change_state() -> None:
    collector = make_collector()
    collector.push({"reward": [2.0, 4.0]})

    first = collector.peek()
    second = collector.peek()

    assert first == second == {"reward": pytest.approx(3.0)}


def test_reset_discards_accumulated_values() -> None:
    collector = make_collector()
    collector.push({"reward": 4.0, "cause": 1})

    collector.reset()

    assert collector.peek() == {}


def test_flush_logs_in_order_then_resets() -> None:
    calls: list[tuple[str, dict[str, float], int]] = []

    def first(metrics: Mapping[str, float], *, step: int) -> None:
        calls.append(("first", dict(metrics), step))

    def second(metrics: Mapping[str, float], *, step: int) -> None:
        calls.append(("second", dict(metrics), step))

    collector = make_collector(loggers=[first, second])
    collector.push({"reward": [3.0, 5.0]})

    flushed = collector.flush(step=42)

    assert flushed == {"reward": pytest.approx(4.0)}
    assert calls == [
        ("first", {"reward": pytest.approx(4.0)}, 42),
        ("second", {"reward": pytest.approx(4.0)}, 42),
    ]
    assert collector.peek() == {}


def test_flush_without_loggers_returns_snapshot_and_resets() -> None:
    collector = make_collector()
    collector.push({"items": 3})

    assert collector.flush(step=1) == {"items_total": pytest.approx(3.0)}
    assert collector.peek() == {}


def test_flush_failure_preserves_state() -> None:
    calls: list[str] = []

    def failing(metrics: Mapping[str, float], *, step: int) -> None:
        calls.append("failing")
        raise RuntimeError("logger unavailable")

    def later(metrics: Mapping[str, float], *, step: int) -> None:
        calls.append("later")

    collector = make_collector(loggers=[failing, later])
    collector.push({"reward": 7.0})

    with pytest.raises(RuntimeError, match="logger unavailable"):
        collector.flush(step=3)

    assert calls == ["failing"]
    assert collector.peek() == {"reward": pytest.approx(7.0)}


def test_flush_with_no_metrics_does_not_call_loggers() -> None:
    calls: list[int] = []
    collector = make_collector(loggers=[lambda metrics, *, step: calls.append(step)])

    assert collector.flush(step=5) == {}
    assert calls == []


def test_push_ignores_unknown_keys_and_rejects_invalid_categorical_values() -> None:
    collector = make_collector()

    collector.push({"unknown": 1.0, "reward": 3.0})

    assert collector.peek() == {"reward": pytest.approx(3.0)}
    with pytest.raises(ValueError, match="unknown values"):
        collector.push({"cause": 9})
    with pytest.raises(ValueError, match="integer-valued"):
        collector.push({"cause": 1.5})


def test_empty_values_are_ignored() -> None:
    collector = make_collector()

    collector.push({"reward": torch.empty(0), "cause": torch.empty(0, dtype=torch.int64)})

    assert collector.peek() == {}


def test_push_accepts_a_tensordict() -> None:
    collector = make_collector()
    values = TensorDict(
        {
            "reward": torch.tensor([2.0, 6.0]),
            "cause": torch.tensor([1, 2]),
        },
        batch_size=[2],
    )

    collector.push(values)

    assert collector.peek() == {
        "reward": pytest.approx(4.0),
        "end_cause_TIMEOUT": pytest.approx(1.0),
        "end_cause_COMPLETED": pytest.approx(1.0),
    }


def test_push_ignores_unknown_tensordict_keys() -> None:
    collector = make_collector()
    values = TensorDict(
        {"other": torch.tensor([1.0]), "reward": torch.tensor([5.0])},
        batch_size=[1],
    )

    collector.push(values)

    assert collector.peek() == {"reward": pytest.approx(5.0)}
