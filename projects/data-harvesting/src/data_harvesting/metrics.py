from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from rl_core import CategoricalMetricSpec, ScalarMetricSpec
from tensordict import TensorDictBase


MetricSpec = ScalarMetricSpec | CategoricalMetricSpec


def extract_terminal_metric_values(
    transitions: TensorDictBase,
    specs: Iterable[MetricSpec],
) -> dict[str, torch.Tensor]:
    """Extract configured environment metrics from completed transitions."""

    done = transitions.get(("next", "done")).reshape(-1).to(torch.bool)
    if not bool(done.any()):
        return {}

    info = transitions.get(("next", "agents", "info"))[done, 0].detach()
    return {spec.key: info.get(spec.key) for spec in specs}


def extract_selected_terminal_metric_values(
    terminal_transitions: TensorDictBase,
    specs: Iterable[MetricSpec],
) -> Mapping[str, torch.Tensor]:
    """Extract metrics after rl-core Evaluator has selected terminal rows."""

    info = terminal_transitions.get(("next", "agents", "info"))
    return {spec.key: info.get(spec.key)[..., 0] for spec in specs}
