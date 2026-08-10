import pytest
import torch
from rl_core import MetricsCollector
from tensordict import TensorDict

from data_harvesting.environment import make_metrics_spec
from data_harvesting.metrics import (
    extract_selected_terminal_metric_values,
    extract_terminal_metric_values,
)


METRIC_KEYS = tuple(metric.key for metric in make_metrics_spec())


def _make_batch(done: list[bool], info_rows: list[dict[str, float]]) -> TensorDict:
    length = len(done)
    info_tensors = {
        key: torch.tensor([[row.get(key, 0.0)] for row in info_rows], dtype=torch.float32)
        for key in METRIC_KEYS
    }
    return TensorDict(
        {
            "next": TensorDict(
                {
                    "done": torch.tensor(done, dtype=torch.bool).view(length, 1),
                    "agents": TensorDict(
                        {"info": TensorDict(info_tensors, batch_size=[length, 1])},
                        batch_size=[length],
                    ),
                },
                batch_size=[length],
            )
        },
        batch_size=[length],
    )


def test_project_extractor_and_rl_core_collector_aggregate_terminal_metrics() -> None:
    specs = make_metrics_spec()
    collector = MetricsCollector(specs=specs)
    batch = _make_batch(
        [False, True, True],
        [
            {},
            {"avg_reward": 2.0, "completion_time": 12.0, "cause": 2.0},
            {"avg_reward": 6.0, "completion_time": 16.0, "cause": 3.0},
        ],
    )

    collector.push(extract_terminal_metric_values(batch, specs))

    metrics = collector.peek()
    assert metrics["avg_reward"] == pytest.approx(4.0)
    assert metrics["completion_time"] == pytest.approx(14.0)
    assert metrics["end_cause_ALL_COLLECTED"] == pytest.approx(1.0)
    assert metrics["end_cause_STALLED"] == pytest.approx(1.0)


def test_project_extractor_ignores_non_terminal_batches() -> None:
    specs = make_metrics_spec()
    batch = _make_batch([False, False], [{"avg_reward": 1.0}, {"avg_reward": 2.0}])

    assert extract_terminal_metric_values(batch, specs) == {}


def test_evaluator_extractor_accepts_preselected_terminal_transitions() -> None:
    specs = make_metrics_spec()
    batch = _make_batch(
        [False, True],
        [{}, {"avg_reward": 3.0, "completion_time": 9.0, "cause": 2.0}],
    )
    terminal = batch[batch.get(("next", "done")).squeeze(-1)]

    values = extract_selected_terminal_metric_values(terminal, specs)

    assert values["avg_reward"].tolist() == [3.0]
    assert values["completion_time"].tolist() == [9.0]
    assert values["cause"].tolist() == [2.0]


def test_evaluator_extractor_selects_one_agent_slot_per_terminal_transition() -> None:
    specs = make_metrics_spec()
    num_transitions = 2
    num_agent_slots = 8
    info_tensors = {
        key: torch.zeros(num_transitions, num_agent_slots, dtype=torch.float32)
        for key in METRIC_KEYS
    }
    info_tensors["avg_reward"] = torch.tensor(
        [
            [3.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            [7.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0],
        ]
    )
    info_tensors["cause"] = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    terminal = TensorDict(
        {
            "next": TensorDict(
                {
                    "agents": TensorDict(
                        {
                            "info": TensorDict(
                                info_tensors,
                                batch_size=[num_transitions],
                            )
                        },
                        batch_size=[num_transitions],
                    )
                },
                batch_size=[num_transitions],
            )
        },
        batch_size=[num_transitions],
    )

    values = extract_selected_terminal_metric_values(terminal, specs)

    assert values["avg_reward"].tolist() == [3.0, 7.0]
    assert values["cause"].tolist() == [2.0, 3.0]
