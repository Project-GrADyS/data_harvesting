from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from tensordict import TensorDict

import rl_core
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


def test_metrics_collector_is_exported_from_package_root() -> None:
    assert rl_core.MetricsCollector is MetricsCollector


def test_collector_requires_at_least_one_spec() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MetricsCollector(specs=[])


def test_collector_rejects_unknown_spec_types() -> None:
    with pytest.raises(TypeError, match="metric specification"):
        MetricsCollector(specs=[object()])  # type: ignore[list-item]


def test_collector_rejects_non_callable_loggers() -> None:
    with pytest.raises(TypeError, match="logger"):
        make_collector(loggers=[object()])


def test_collector_snapshots_spec_and_logger_iterables() -> None:
    calls: list[int] = []
    spec = ScalarMetricSpec(key="reward", reducer=ScalarReducer.MEAN)
    collector = MetricsCollector(
        specs=(item for item in [spec]),
        loggers=(logger for logger in [lambda metrics, *, step: calls.append(step)]),
    )

    collector.push({"reward": 2.0})
    collector.flush(step=4)

    assert calls == [4]


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


def test_scalar_reducers_flatten_arbitrary_sample_dimensions() -> None:
    collector = make_collector()

    collector.push(
        {
            "reward": torch.tensor([[1.0, 2.0], [3.0, 6.0]]),
            "items": torch.tensor([[1, 2], [3, 4]]),
        }
    )

    assert collector.peek() == {
        "reward": pytest.approx(3.0),
        "items_total": pytest.approx(10.0),
    }


def test_categorical_snapshot_includes_zero_counts_after_metric_is_seen() -> None:
    collector = make_collector()

    collector.push({"cause": [1, 1]})

    assert collector.peek() == {
        "end_cause_TIMEOUT": 2.0,
        "end_cause_COMPLETED": 0.0,
    }


def test_categorical_metric_accepts_integer_valued_floats() -> None:
    collector = make_collector()

    collector.push({"cause": torch.tensor([1.0, 2.0, 2.0])})

    assert collector.peek() == {
        "end_cause_TIMEOUT": 1.0,
        "end_cause_COMPLETED": 2.0,
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


def test_each_logger_receives_an_independent_snapshot() -> None:
    calls: list[dict[str, float]] = []

    def mutating(metrics: Mapping[str, float], *, step: int) -> None:
        metrics["reward"] = 999.0  # type: ignore[index]
        calls.append(dict(metrics))

    def observing(metrics: Mapping[str, float], *, step: int) -> None:
        calls.append(dict(metrics))

    collector = make_collector(loggers=[mutating, observing])
    collector.push({"reward": 3.0})

    collector.flush(step=1)

    assert calls == [{"reward": 999.0}, {"reward": 3.0}]


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


def test_later_logger_failure_also_preserves_state() -> None:
    calls: list[str] = []

    def first(metrics: Mapping[str, float], *, step: int) -> None:
        calls.append("first")

    def failing(metrics: Mapping[str, float], *, step: int) -> None:
        calls.append("failing")
        raise RuntimeError("logger unavailable")

    collector = make_collector(loggers=[first, failing])
    collector.push({"reward": 7.0})

    with pytest.raises(RuntimeError, match="logger unavailable"):
        collector.flush(step=3)

    assert calls == ["first", "failing"]
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


def test_unknown_categorical_values_are_reported_once_in_sorted_order() -> None:
    collector = make_collector()

    with pytest.raises(ValueError, match=r"\[-2, 9\]"):
        collector.push({"cause": [9, -2, 9]})


@pytest.mark.parametrize("values", [None, [1.0], "reward", 3.0])
def test_push_requires_mapping_or_tensordict(values: object) -> None:
    collector = make_collector()

    with pytest.raises(TypeError, match="mapping or a TensorDictBase"):
        collector.push(values)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, "1.0", object()])
def test_scalar_metric_rejects_non_numeric_python_values(value: object) -> None:
    collector = make_collector()

    with pytest.raises(TypeError, match="real numeric"):
        collector.push({"reward": value})  # type: ignore[dict-item]


@pytest.mark.parametrize("value", [["1.0"], [object()]])
def test_scalar_metric_normalizes_invalid_sequence_errors_to_type_error(
    value: list[object],
) -> None:
    collector = make_collector()

    with pytest.raises(TypeError, match="real numeric"):
        collector.push({"reward": value})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([True, False]),
        torch.tensor([1 + 2j]),
    ],
)
def test_scalar_metric_rejects_boolean_and_complex_tensors(value: torch.Tensor) -> None:
    collector = make_collector()

    with pytest.raises(TypeError, match="real numeric"):
        collector.push({"reward": value})


def test_scalar_metric_rejects_unsupported_real_tensor_dtype() -> None:
    collector = make_collector()

    with pytest.raises(TypeError, match="numeric values"):
        collector.push({"reward": torch.tensor([1], dtype=torch.uint16)})


@pytest.mark.parametrize(
    "value",
    [
        True,
        torch.tensor([True, False]),
        torch.tensor([1 + 0j]),
    ],
)
def test_categorical_metric_rejects_boolean_and_complex_values(value: object) -> None:
    collector = make_collector()

    with pytest.raises(TypeError, match="real numeric"):
        collector.push({"cause": value})  # type: ignore[dict-item]


def test_categorical_metric_rejects_unsupported_integer_tensor_dtype() -> None:
    collector = make_collector()

    with pytest.raises(TypeError, match="integer values"):
        collector.push({"cause": torch.tensor([1], dtype=torch.uint16)})


def test_empty_values_are_ignored() -> None:
    collector = make_collector()

    collector.push({"reward": torch.empty(0), "cause": torch.empty(0, dtype=torch.int64)})

    assert collector.peek() == {}


def test_pushed_tensors_are_detached_moved_to_collector_device_and_eagerly_consumed() -> None:
    collector = MetricsCollector(
        specs=[ScalarMetricSpec(key="reward", reducer=ScalarReducer.SUM)],
        device=torch.device("cpu"),
    )
    source = torch.tensor([1.0, 2.0], requires_grad=True)

    collector.push({"reward": source})
    with torch.no_grad():
        source.add_(100.0)

    assert collector.peek() == {"reward": 3.0}
    assert collector._scalar_totals["reward"].device.type == "cpu"
    assert not collector._scalar_totals["reward"].requires_grad


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


def test_push_ignores_nested_tensordict_leaves_and_reads_top_level_leaves() -> None:
    collector = make_collector()
    values = TensorDict(
        {
            "nested": TensorDict(
                {"reward": torch.tensor([999.0])},
                batch_size=[1],
            ),
            "reward": torch.tensor([5.0]),
        },
        batch_size=[1],
    )

    collector.push(values)

    assert collector.peek() == {"reward": pytest.approx(5.0)}
