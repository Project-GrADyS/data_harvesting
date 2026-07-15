from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest
import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn
from torchrl.envs.utils import ExplorationType, exploration_type

import rl_core
import rl_core.evaluation as evaluation
from rl_core.evaluation import EvaluationConfig, Evaluator, validate_evaluation_config
from rl_core.metrics import MetricsCollector, ScalarMetricSpec, ScalarReducer
from rl_core.scheduling import Scheduler


class Policy(nn.Module):
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict


def make_rollout(
    *,
    score: float,
    terminal: tuple[bool, ...] = (False, True),
    terminal_key: str | tuple[str, ...] = ("next", "done"),
) -> TensorDictBase:
    length = len(terminal)
    return TensorDict(
        {
            terminal_key: torch.tensor(terminal).unsqueeze(-1),
            ("next", "agents", "info", "score"): torch.stack(
                (
                    torch.full((length,), score),
                    torch.full((length,), score + 100),
                ),
                dim=-1,
            ),
            ("next", "observation"): torch.arange(length, dtype=torch.float32).unsqueeze(-1),
        },
        batch_size=[length],
    )


class Environment:
    def __init__(
        self,
        rollouts: list[object],
        policy: Policy,
        *,
        expected_exploration: ExplorationType | None = ExplorationType.MODE,
    ) -> None:
        self._rollouts = iter(rollouts)
        self.policy = policy
        self.expected_exploration = expected_exploration
        self.rollout_calls: list[dict[str, object]] = []
        self.closed = False

    def rollout(self, **kwargs) -> TensorDictBase:
        assert self.policy.training is False
        assert torch.is_grad_enabled() is False
        if self.expected_exploration is not None:
            assert exploration_type() is self.expected_exploration
        self.rollout_calls.append(kwargs)
        result = next(self._rollouts)
        if isinstance(result, BaseException):
            raise result
        return result  # type: ignore[return-value]

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


def make_evaluator(
    *,
    config: EvaluationConfig | None = None,
    environment: Environment | None = None,
    policy: Policy | None = None,
    metrics: MetricsCollector | None = None,
    extractor: Callable[[TensorDictBase], dict[str, torch.Tensor]] = extract_first_agent,
) -> tuple[Evaluator, Policy, Environment]:
    policy = policy or Policy()
    environment = environment or Environment([make_rollout(score=2.0)], policy)
    evaluator = Evaluator(
        config=config or EvaluationConfig(num_episodes=1, max_steps=10),
        env_factory=lambda: environment,  # type: ignore[arg-type]
        policy=policy,
        metrics=metrics or make_metrics(),
        metric_extractor=extractor,
    )
    return evaluator, policy, environment


def test_evaluator_runs_episodes_and_reports_terminal_metrics() -> None:
    policy = Policy()
    logged: list[tuple[dict[str, float], int]] = []
    environment = Environment([make_rollout(score=2.0), make_rollout(score=4.0)], policy)
    config = EvaluationConfig(
        num_episodes=2,
        max_steps=50,
        rollout_kwargs={"break_when_any_done": True},
    )
    evaluator, _, _ = make_evaluator(
        config=config,
        environment=environment,
        policy=policy,
        metrics=make_metrics(logged),
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

    evaluator, _, _ = make_evaluator(environment=environment, policy=policy, extractor=extract)

    evaluator(5)

    assert received[0].batch_size == torch.Size([1])
    assert received[0].get(("next", "observation")).item() == 1.0


def test_all_terminal_transitions_are_passed_to_extractor() -> None:
    policy = Policy()
    environment = Environment(
        [make_rollout(score=3.0, terminal=(True, False, True))],
        policy,
    )
    received: list[TensorDictBase] = []

    def extract(transitions: TensorDictBase) -> dict[str, torch.Tensor]:
        received.append(transitions)
        return extract_first_agent(transitions)

    evaluator, _, _ = make_evaluator(environment=environment, policy=policy, extractor=extract)

    evaluator(1)

    assert received[0].batch_size == torch.Size([2])
    assert received[0].get(("next", "observation")).squeeze(-1).tolist() == [0.0, 2.0]


def test_evaluator_supports_a_top_level_string_terminal_key() -> None:
    policy = Policy()
    environment = Environment([make_rollout(score=3.0, terminal_key="done")], policy)
    evaluator, _, _ = make_evaluator(
        config=EvaluationConfig(num_episodes=1, max_steps=10, terminal_key="done"),
        environment=environment,
        policy=policy,
    )

    assert evaluator.run(0) == {"score": 3.0}


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


def test_config_is_passive_frozen_slotted_and_keyword_only() -> None:
    config = EvaluationConfig(num_episodes=0, max_steps=-1)

    assert config.num_episodes == 0
    assert not hasattr(config, "__dict__")
    with pytest.raises(FrozenInstanceError):
        config.max_steps = 5  # type: ignore[misc]
    with pytest.raises(TypeError):
        EvaluationConfig(1, 10)  # type: ignore[misc]


def test_config_retains_caller_owned_rollout_kwargs() -> None:
    kwargs = {"break_when_all_done": True}
    config = EvaluationConfig(num_episodes=1, max_steps=10, rollout_kwargs=kwargs)

    assert config.rollout_kwargs is kwargs
    kwargs["break_when_all_done"] = False
    assert config.rollout_kwargs["break_when_all_done"] is False


@pytest.mark.parametrize("field", ["num_episodes", "max_steps"])
@pytest.mark.parametrize("value", [None, 1.5, "1", True])
def test_config_integer_fields_reject_wrong_types(field: str, value: object) -> None:
    kwargs: dict[str, object] = {"num_episodes": 1, "max_steps": 10}
    kwargs[field] = value

    with pytest.raises(TypeError, match=field):
        validate_evaluation_config(EvaluationConfig(**kwargs))  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["num_episodes", "max_steps"])
@pytest.mark.parametrize("value", [0, -1])
def test_config_integer_fields_reject_non_positive_values(field: str, value: int) -> None:
    kwargs = {"num_episodes": 1, "max_steps": 10}
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        validate_evaluation_config(EvaluationConfig(**kwargs))


@pytest.mark.parametrize("terminal_key", [None, 3, ["next", "done"], ("next", 3)])
def test_terminal_key_rejects_wrong_types(terminal_key: object) -> None:
    config = EvaluationConfig(num_episodes=1, max_steps=10, terminal_key=terminal_key)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="terminal_key"):
        validate_evaluation_config(config)


@pytest.mark.parametrize("terminal_key", ["", (), ("next", "")])
def test_terminal_key_rejects_empty_values(terminal_key: str | tuple[str, ...]) -> None:
    config = EvaluationConfig(num_episodes=1, max_steps=10, terminal_key=terminal_key)

    with pytest.raises(ValueError, match="terminal_key"):
        validate_evaluation_config(config)


@pytest.mark.parametrize("exploration", ["mode", 1, object()])
def test_exploration_type_rejects_wrong_types(exploration: object) -> None:
    config = EvaluationConfig(
        num_episodes=1,
        max_steps=10,
        exploration_type=exploration,  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="exploration_type"):
        validate_evaluation_config(config)


def test_rollout_kwargs_must_be_a_mapping() -> None:
    config = EvaluationConfig(num_episodes=1, max_steps=10, rollout_kwargs=[])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="rollout_kwargs"):
        validate_evaluation_config(config)


@pytest.mark.parametrize("reserved", ["max_steps", "policy"])
def test_rollout_kwargs_reject_evaluator_owned_arguments(reserved: str) -> None:
    config = EvaluationConfig(num_episodes=1, max_steps=10, rollout_kwargs={reserved: object()})

    with pytest.raises(ValueError, match="evaluator-owned"):
        validate_evaluation_config(config)


def test_config_accepts_string_and_nested_terminal_keys() -> None:
    validate_evaluation_config(EvaluationConfig(num_episodes=1, max_steps=10, terminal_key="done"))
    validate_evaluation_config(
        EvaluationConfig(num_episodes=1, max_steps=10, terminal_key=("next", "agents", "done"))
    )


def test_evaluator_invokes_standalone_config_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EvaluationConfig(num_episodes=1, max_steps=10)
    calls: list[EvaluationConfig] = []
    original = evaluation.validate_evaluation_config

    def validate(value: EvaluationConfig) -> None:
        calls.append(value)
        original(value)

    monkeypatch.setattr(evaluation, "validate_evaluation_config", validate)
    make_evaluator(config=config)

    assert calls == [config]


def test_evaluator_rejects_invalid_passive_config_at_use_boundary() -> None:
    with pytest.raises(ValueError, match="num_episodes"):
        make_evaluator(config=EvaluationConfig(num_episodes=0, max_steps=10))


@pytest.mark.parametrize("step", [None, 1.5, "1", True])
def test_step_rejects_wrong_types(step: object) -> None:
    evaluator, _, _ = make_evaluator()

    with pytest.raises(TypeError, match="step"):
        evaluator.run(step)  # type: ignore[arg-type]


def test_step_rejects_negative_values() -> None:
    evaluator, _, _ = make_evaluator()

    with pytest.raises(ValueError, match="step"):
        evaluator.run(-1)


@pytest.mark.parametrize("configured", [ExplorationType.RANDOM, ExplorationType.MODE])
def test_configured_exploration_context_is_used(configured: ExplorationType) -> None:
    policy = Policy()
    environment = Environment([make_rollout(score=1.0)], policy, expected_exploration=configured)
    evaluator, _, _ = make_evaluator(
        config=EvaluationConfig(num_episodes=1, max_steps=10, exploration_type=configured),
        policy=policy,
        environment=environment,
    )

    evaluator.run(0)


def test_none_exploration_preserves_callers_context() -> None:
    policy = Policy()
    environment = Environment(
        [make_rollout(score=1.0)],
        policy,
        expected_exploration=ExplorationType.RANDOM,
    )
    evaluator, _, _ = make_evaluator(
        config=EvaluationConfig(num_episodes=1, max_steps=10, exploration_type=None),
        policy=policy,
        environment=environment,
    )

    with evaluation.set_exploration_type(ExplorationType.RANDOM):
        evaluator.run(0)


@pytest.mark.parametrize("initial_training", [True, False])
@pytest.mark.parametrize("failure_point", ["rollout", "extractor", "flush"])
def test_policy_mode_and_environment_are_restored_after_failure(
    initial_training: bool,
    failure_point: str,
) -> None:
    policy = Policy().train(initial_training)
    rollout_or_error: TensorDictBase | BaseException = make_rollout(score=1.0)
    if failure_point == "rollout":
        rollout_or_error = RuntimeError("rollout failed")
    environment = Environment([rollout_or_error], policy)
    metrics = make_metrics()

    def extractor(transitions: TensorDictBase) -> dict[str, torch.Tensor]:
        if failure_point == "extractor":
            raise RuntimeError("extractor failed")
        return extract_first_agent(transitions)

    if failure_point == "flush":
        def fail_flush(values: dict[str, float], *, step: int) -> None:
            raise RuntimeError("flush failed")

        metrics._loggers = (fail_flush,)

    evaluator, _, _ = make_evaluator(
        environment=environment,
        policy=policy,
        metrics=metrics,
        extractor=extractor,
    )

    with pytest.raises(RuntimeError, match=f"{failure_point} failed"):
        evaluator(2)

    assert policy.training is initial_training
    assert environment.closed is True


def test_rollout_must_be_a_tensordict() -> None:
    policy = Policy()
    environment = Environment([{"next": {"done": torch.tensor([True])}}], policy)
    evaluator, _, _ = make_evaluator(environment=environment, policy=policy)

    with pytest.raises(TypeError, match="TensorDictBase"):
        evaluator.run(0)


def test_terminal_key_must_exist() -> None:
    policy = Policy()
    rollout = make_rollout(score=1.0)
    evaluator, _, _ = make_evaluator(
        config=EvaluationConfig(num_episodes=1, max_steps=2, terminal_key=("next", "missing")),
        environment=Environment([rollout], policy),
        policy=policy,
    )

    with pytest.raises(KeyError, match="terminal_key"):
        evaluator.run(0)


def test_terminal_key_must_resolve_to_one_value_per_transition() -> None:
    policy = Policy()
    rollout = make_rollout(score=1.0)
    rollout.set(("next", "bad_done"), torch.ones(2, 2, dtype=torch.bool))
    evaluator, _, _ = make_evaluator(
        config=EvaluationConfig(num_episodes=1, max_steps=2, terminal_key=("next", "bad_done")),
        environment=Environment([rollout], policy),
        policy=policy,
    )

    with pytest.raises(ValueError, match="one Boolean value per transition|expected rollout batch size"):
        evaluator.run(1)


def test_rollout_requires_at_least_one_terminal_transition() -> None:
    policy = Policy()
    environment = Environment([make_rollout(score=1.0, terminal=(False, False))], policy)
    evaluator, _, _ = make_evaluator(environment=environment, policy=policy)

    with pytest.raises(RuntimeError, match="without a terminal transition"):
        evaluator(2)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("config", object(), "config"),
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


def test_evaluation_api_is_exported_from_module_and_package_root() -> None:
    assert evaluation.__all__ == [
        "EvaluationConfig",
        "Evaluator",
        "TerminalMetricExtractor",
        "validate_evaluation_config",
    ]
    assert rl_core.EvaluationConfig is EvaluationConfig
    assert rl_core.Evaluator is Evaluator
    assert rl_core.TerminalMetricExtractor is evaluation.TerminalMetricExtractor
    assert rl_core.validate_evaluation_config is validate_evaluation_config
