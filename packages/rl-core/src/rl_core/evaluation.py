from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias

import torch
from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torch import nn
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

from .metrics.collector import MetricValues, MetricsCollector


EnvironmentFactory: TypeAlias = Callable[[], EnvBase]
TerminalMetricExtractor: TypeAlias = Callable[[TensorDictBase], MetricValues]


_RESERVED_ROLLOUT_KWARGS = {"max_steps", "policy"}


def _validate_positive_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _validate_non_negative_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")


def _validate_nested_key(name: str, key: object) -> None:
    valid_string = isinstance(key, str) and bool(key)
    valid_tuple = (
        isinstance(key, tuple)
        and bool(key)
        and all(isinstance(part, str) and bool(part) for part in key)
    )
    if not valid_string and not valid_tuple:
        raise ValueError(f"{name} must be a non-empty string or tuple of non-empty strings.")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationConfig:
    """Configuration for a finite TorchRL policy evaluation."""

    num_episodes: int
    max_steps: int
    terminal_key: NestedKey = ("next", "done")
    exploration_type: ExplorationType | None = ExplorationType.MODE
    rollout_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_positive_integer("num_episodes", self.num_episodes)
        _validate_positive_integer("max_steps", self.max_steps)
        _validate_nested_key("terminal_key", self.terminal_key)
        if self.exploration_type is not None and not isinstance(self.exploration_type, ExplorationType):
            raise TypeError("exploration_type must be an ExplorationType or None.")
        if not isinstance(self.rollout_kwargs, Mapping):
            raise TypeError("rollout_kwargs must be a mapping.")

        rollout_kwargs = dict(self.rollout_kwargs)
        conflicts = _RESERVED_ROLLOUT_KWARGS.intersection(rollout_kwargs)
        if conflicts:
            raise ValueError(f"rollout_kwargs contains evaluator-owned arguments: {sorted(conflicts)}")
        object.__setattr__(self, "rollout_kwargs", MappingProxyType(rollout_kwargs))


class Evaluator:
    """Run finite policy evaluations and report terminal metrics."""

    def __init__(
        self,
        *,
        config: EvaluationConfig,
        env_factory: EnvironmentFactory,
        policy: nn.Module,
        metrics: MetricsCollector,
        metric_extractor: TerminalMetricExtractor,
    ) -> None:
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be an EvaluationConfig, got {type(config)}.")
        if not callable(env_factory):
            raise TypeError("env_factory must be callable.")
        if not isinstance(policy, nn.Module):
            raise TypeError("policy must be a torch.nn.Module.")
        if not isinstance(metrics, MetricsCollector):
            raise TypeError("metrics must be a MetricsCollector.")
        if not callable(metric_extractor):
            raise TypeError("metric_extractor must be callable.")

        self.config = config
        self.env_factory = env_factory
        self.policy = policy
        self.metrics = metrics
        self.metric_extractor = metric_extractor

    def __call__(self, step: int) -> dict[str, float]:
        """Run an evaluation, making the evaluator directly usable as a scheduled callback."""

        return self.run(step)

    def run(self, step: int) -> dict[str, float]:
        """Evaluate the policy and flush aggregated metrics at ``step``."""

        _validate_non_negative_integer("step", step)
        self.metrics.reset()
        environment = self.env_factory()
        policy_was_training = self.policy.training

        try:
            self.policy.eval()
            exploration_context = (
                nullcontext()
                if self.config.exploration_type is None
                else set_exploration_type(self.config.exploration_type)
            )
            with torch.no_grad(), exploration_context:
                for _ in range(self.config.num_episodes):
                    rollout = environment.rollout(
                        max_steps=self.config.max_steps,
                        policy=self.policy,
                        **self.config.rollout_kwargs,
                    )
                    terminal_transitions = self._terminal_transitions(rollout)
                    self.metrics.push(self.metric_extractor(terminal_transitions))

            return self.metrics.flush(step=step)
        finally:
            self.policy.train(policy_was_training)
            environment.close()

    def _terminal_transitions(self, rollout: TensorDictBase) -> TensorDictBase:
        if not isinstance(rollout, TensorDictBase):
            raise TypeError("environment.rollout() must return a TensorDictBase.")

        if self.config.terminal_key not in rollout.keys(include_nested=True):
            raise KeyError(
                f"terminal_key {self.config.terminal_key!r} not found in rollout keys: {list(rollout.keys())}"
            )

        terminal = rollout.get(self.config.terminal_key).to(torch.bool)
        while terminal.ndim > rollout.ndim:
            if terminal.shape[-1] != 1:
                raise ValueError(
                    f"terminal_key {self.config.terminal_key!r} must identify one Boolean value per transition."
                )
            terminal = terminal.squeeze(-1)

        if tuple(terminal.shape) != tuple(rollout.batch_size):
            raise ValueError(
                f"terminal_key {self.config.terminal_key!r} has shape {tuple(terminal.shape)}, "
                f"expected rollout batch size {tuple(rollout.batch_size)}."
            )
        if not bool(terminal.any()):
            raise RuntimeError(
                f"Evaluation rollout reached max_steps={self.config.max_steps} without a terminal transition."
            )
        return rollout[terminal]


__all__ = ["EvaluationConfig", "Evaluator", "TerminalMetricExtractor"]
