from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn
from torchrl.envs.utils import ExplorationType, exploration_type

import rl_core
from rl_core.evaluation import EvaluationConfig, Evaluator
from rl_core.metrics import MetricsCollector, ScalarMetricSpec, ScalarReducer
from rl_core.scheduling import Scheduler


class Policy(nn.Module):
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict


def make_rollout(*, score: float, terminal: bool = True) -> TensorDictBase:
    return TensorDict(
        {
            ("next", "done"): torch.tensor([[False], [terminal]]),
            ("next", "agents", "info", "score"): torch.tensor(
                [[-1.0, -2.0], [score, score + 100.0]]
            ),
            ("next", "observation"): torch.tensor([[10.0], [20.0]]),
        },
        batch_size=[2],
    )


class Environment:
    def __init__(self, rollouts: list[TensorDictBase], policy: Policy) -> None:
        self._rollouts = iter(rollouts)
        self.policy = policy
        self.rollout_calls: list[dict[str, object]] = []
        self.closed = False

    def rollout(self, **kwargs) -> TensorDictBase:
        assert self.policy.training is False
        assert torch.is_grad_enabled() is False
        assert exploration_type() is ExplorationType.MODE
        self.rollout_calls.append(kwargs)
        return next(self._rollouts)

    def close(self) -> None:
        self.closed = True


def make_metrics(logged: list[tuple[dict[str, float], int]] | None = None) -> MetricsCollector:
    loggers = []
    if logged is not None:
        loggers.append(lambda values, *, step: logged.append((values, step)))
    return MetricsCollector(
        specs=[ScalarMetricSpec(key="score", reducer=ScalarReducer.MEAN)],
        loggers=loggers,
    )


def extract_first_agent(terminal_transitions: TensorDictBase) -> dict[str, torch.Tensor]:
    score = terminal_transitions.get(("next", "agents", "info", "score"))
    return {"score": score[..., 0]}


def test_evaluator_runs_episodes_and_reports_terminal_metrics() -> None:
    policy = Policy()
    logged: list[tuple[dict[str, float], int]] = []
    metrics = make_metrics(logged)
    environment = Environment([make_rollout(score=2.0), make_rollout(score=4.0)], policy)
    config = EvaluationConfig(
        num_episodes=2,
        max_steps=50,
        rollout_kwargs={"break_when_any_done": True},
    )
    evaluator = Evaluator(
        config=config,
        env_factory=lambda: environment,  # type: ignore[arg-type]
        policy=policy,
        metrics=metrics,
        metric_extractor=extract_first_agent,
    )

    result = evaluator.run(1_000)

    assert result == {"score": 3.0}
    assert logged == [({"score": 3.0}, 1_000)]
    assert environment.closed is True
    assert len(environment.rollout_calls) == 2
    assert environment.rollout_calls[0] == {
        "max_steps": 50,
        "policy": policy,
        "break_when_any_done": True,
    }
    assert policy.training is True


def test_extractor_receives_complete_terminal_transitions_only() -> None:
    policy = Policy()
    environment = Environment([make_rollout(score=7.0)], policy)
    received: list[TensorDictBase] = []

    def extract(terminal_transitions: TensorDictBase) -> dict[str, torch.Tensor]:
        received.append(terminal_transitions)
        assert ("next", "observation") in terminal_transitions.keys(include_nested=True)
        return extract_first_agent(terminal_transitions)

    evaluator = Evaluator(
        config=EvaluationConfig(num_episodes=1, max_steps=10),
        env_factory=lambda: environment,  # type: ignore[arg-type]
        policy=policy,
        metrics=make_metrics(),
        metric_extractor=extract,
    )

    evaluator(5)

    assert received[0].batch_size == torch.Size([1])
    assert received[0].get(("next", "observation")).item() == 20.0


def test_evaluator_is_compatible_with_scheduler() -> None:
    policy = Policy()
    environments: list[Environment] = []

    def make_environment() -> Environment:
        environment = Environment([make_rollout(score=6.0)], policy)
        environments.append(environment)
        return environment

    logged: list[tuple[dict[str, float], int]] = []
    evaluator = Evaluator(
        config=EvaluationConfig(num_episodes=1, max_steps=10),
        env_factory=make_environment,  # type: ignore[arg-type]
        policy=policy,
        metrics=make_metrics(logged),
        metric_extractor=extract_first_agent,
    )
    scheduler = Scheduler()
    scheduler.register("evaluation", every=10, callback=evaluator)

    scheduler.step(increment=10)

    assert logged == [({"score": 6.0}, 10)]
    assert environments[0].closed is True


@pytest.mark.parametrize("initial_training", [True, False])
def test_policy_mode_and_environment_are_restored_after_failure(initial_training: bool) -> None:
    policy = Policy().train(initial_training)
    environment = Environment([make_rollout(score=1.0, terminal=False)], policy)
    evaluator = Evaluator(
        config=EvaluationConfig(num_episodes=1, max_steps=2),
        env_factory=lambda: environment,  # type: ignore[arg-type]
        policy=policy,
        metrics=make_metrics(),
        metric_extractor=extract_first_agent,
    )

    with pytest.raises(RuntimeError, match="without a terminal transition"):
        evaluator(2)

    assert policy.training is initial_training
    assert environment.closed is True


def test_evaluator_resets_metrics_before_each_invocation() -> None:
    policy = Policy()
    environments = iter(
        [
            Environment([make_rollout(score=2.0)], policy),
            Environment([make_rollout(score=8.0)], policy),
        ]
    )
    evaluator = Evaluator(
        config=EvaluationConfig(num_episodes=1, max_steps=2),
        env_factory=lambda: next(environments),  # type: ignore[arg-type]
        policy=policy,
        metrics=make_metrics(),
        metric_extractor=extract_first_agent,
    )

    assert evaluator(1) == {"score": 2.0}
    assert evaluator(2) == {"score": 8.0}


def test_config_validates_and_defensively_copies_rollout_kwargs() -> None:
    kwargs = {"break_when_all_done": True}
    config = EvaluationConfig(num_episodes=1, max_steps=10, rollout_kwargs=kwargs)
    kwargs["break_when_all_done"] = False

    assert config.rollout_kwargs["break_when_all_done"] is True

    with pytest.raises(ValueError, match="num_episodes"):
        EvaluationConfig(num_episodes=0, max_steps=10)
    with pytest.raises(ValueError, match="max_steps"):
        EvaluationConfig(num_episodes=1, max_steps=0)
    with pytest.raises(ValueError, match="terminal_key"):
        EvaluationConfig(num_episodes=1, max_steps=10, terminal_key=())
    with pytest.raises(ValueError, match="evaluator-owned"):
        EvaluationConfig(num_episodes=1, max_steps=10, rollout_kwargs={"policy": object()})


def test_config_allows_project_selected_terminal_key_and_exploration_behavior() -> None:
    config = EvaluationConfig(
        num_episodes=1,
        max_steps=10,
        terminal_key=("next", "agents", "done"),
        exploration_type=None,
    )

    assert config.terminal_key == ("next", "agents", "done")
    assert config.exploration_type is None
    assert not hasattr(config, "seed")


def test_terminal_key_must_resolve_to_one_value_per_transition() -> None:
    policy = Policy()
    rollout = make_rollout(score=1.0)
    rollout.set(("next", "bad_done"), torch.ones(2, 2, dtype=torch.bool))
    environment = Environment([rollout], policy)
    evaluator = Evaluator(
        config=EvaluationConfig(num_episodes=1, max_steps=2, terminal_key=("next", "bad_done")),
        env_factory=lambda: environment,  # type: ignore[arg-type]
        policy=policy,
        metrics=make_metrics(),
        metric_extractor=extract_first_agent,
    )

    with pytest.raises(ValueError, match="one Boolean value per transition"):
        evaluator(1)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("env_factory", None, "env_factory"),
        ("policy", object(), "policy"),
        ("metrics", object(), "metrics"),
        ("metric_extractor", None, "metric_extractor"),
    ],
)
def test_evaluator_validates_collaborators(argument: str, value: object, message: str) -> None:
    arguments: dict[str, object] = {
        "config": EvaluationConfig(num_episodes=1, max_steps=2),
        "env_factory": lambda: None,
        "policy": Policy(),
        "metrics": make_metrics(),
        "metric_extractor": extract_first_agent,
    }
    arguments[argument] = value

    with pytest.raises(TypeError, match=message):
        Evaluator(**arguments)  # type: ignore[arg-type]


def test_evaluation_api_is_exported_from_package_root() -> None:
    assert rl_core.EvaluationConfig is EvaluationConfig
    assert rl_core.Evaluator is Evaluator
