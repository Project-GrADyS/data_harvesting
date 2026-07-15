# RL Core

`rl-core` contains small, configurable primitives shared by reinforcement-learning projects.

## Metrics

The `rl_core.metrics` module aggregates named tensors or numeric values according to declarative metric specifications.
Projects remain responsible for extracting values from their own batches. The package includes console and optional
MLflow loggers, and callers can provide any callable with the same interface.

```python
from rl_core.metrics import ConsoleMetricLogger, MetricsCollector, ScalarMetricSpec, ScalarReducer


collector = MetricsCollector(
    specs=[ScalarMetricSpec(key="reward", reducer=ScalarReducer.MEAN)],
    loggers=[ConsoleMetricLogger(prefix="train")],
)
collector.push({"reward": [1.0, 2.0, 3.0]})
collector.flush(step=100)
```

`push()` also accepts a `TensorDictBase` whose top-level keys match configured metric keys. TensorDict is a mandatory
package dependency and needs no extra installation option.

`MLflowMetricLogger` logs to the currently active MLflow run and is available through the `mlflow` extra:

```python
from rl_core.metrics import MLflowMetricLogger

logger = MLflowMetricLogger(prefix="evaluation")
```

```bash
pip install "rl-core[mlflow]"
```

The collector exposes four state operations:

- `push(values)` accumulates values.
- `peek()` returns the current aggregates without changing state.
- `flush(step)` sends the current aggregates to every logger and resets after successful logging.
- `reset()` discards accumulated values.

## Collection

`rl_core.collection` provides a typed wrapper over TorchRL's synchronous and asynchronous, single- and multi-worker
collectors. It selects the appropriate collector and guarantees shutdown while leaving iteration, learning, and policy
weight-update timing to the project.

```python
from rl_core.collection import CollectionMode, CollectorConfig, make_collector

config = CollectorConfig(
    mode=CollectionMode.SYNC,
    frames_per_batch=1_024,
    total_frames=100_352,
    num_workers=2,
    device="cpu",
)

with make_collector(config=config, env_factory=make_env, policy=exploration_policy) as collector:
    for batch in collector:
        algorithm.learn(batch)
        collector.update_policy_weights_()
```

Use `collector_kwargs` for supported TorchRL options that are not represented directly by `CollectorConfig`. Wrapper-owned
constructor arguments cannot be overridden through that mapping. The mapping remains caller-owned and is passed through
without copying. As required by Python multiprocessing, construct
multi-worker collectors under an `if __name__ == "__main__"` guard.

## Checkpointing

`rl_core.checkpointing` stores project-created state dictionaries in a versioned checkpoint envelope. Projects decide
what to save and when; the package handles persistence, validation, retention, and restoration.

```python
from rl_core.checkpointing import Checkpoint, CheckpointManager, LocalCheckpointStore

local_store = LocalCheckpointStore("checkpoints", keep_last=3)
manager = CheckpointManager([local_store])

manager.save(
    Checkpoint(
        step=experience_steps,
        state={
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        metadata={"algorithm": "maddpg"},
    )
)

checkpoint = local_store.load_latest()
policy.load_state_dict(checkpoint.state["policy"])
optimizer.load_state_dict(checkpoint.state["optimizer"])
```

`LocalCheckpointStore` writes atomically and can retain only the latest checkpoints. `MLflowCheckpointStore` saves and
loads the same envelope from run artifacts and is available through the existing `mlflow` extra. Checkpoint files use
Python pickle through PyTorch and must only be loaded from trusted sources.

Checkpoint `state` and `metadata` mappings remain caller-owned. Saving sends the current contents to the selected store
without introducing a read-only proxy or defensive clone.

## Scheduling

`rl_core.scheduling` dispatches named callbacks at fixed training-step intervals. The scheduler owns only its logical
clock and callback cadence; projects retain control over the training loop and the work performed by each callback.

```python
from rl_core.scheduling import Scheduler


scheduler = Scheduler()
scheduler.register("metrics", every=1_000, callback=log_metrics)
scheduler.register("checkpoint", every=10_000, callback=save_checkpoint)
scheduler.register("evaluation", every=25_000, callback=run_evaluation)

for batch in collector:
    algorithm.learn(batch)
    scheduler.step()
```

Callbacks receive the resulting current step and run synchronously in registration order. `step(increment=...)` can
advance by a collector batch containing multiple frames. Crossing one or more occurrences of an interval invokes that
callback once at the resulting step:

```python
scheduler.step(increment=batch.numel())
```

If a callback raises, dispatch stops and the error propagates. The clock remains advanced and missed callbacks are not
retried. Registrations added or removed during dispatch affect the following call to `step()`.

The clock can be included in a checkpoint while callback registrations remain application code:

```python
checkpoint_state["scheduler"] = scheduler.state_dict()
scheduler.load_state_dict(checkpoint_state["scheduler"])
```

## Evaluation

`rl_core.evaluation` runs a policy for a finite number of episodes and reports terminal metrics through a dedicated
`MetricsCollector`. It is callable with a training step, so scheduling evaluation requires no evaluation-specific
condition in the training loop.

```python
from rl_core.evaluation import EvaluationConfig, Evaluator


def extract_metrics(terminal_transitions):
    info = terminal_transitions.get(("next", "agents", "info"))
    return {
        "episode_reward": info.get("episode_reward")[..., 0],
        "completion_time": info.get("completion_time")[..., 0],
    }


evaluator = Evaluator(
    config=EvaluationConfig(num_episodes=10, max_steps=1_000),
    env_factory=make_evaluation_env,
    policy=policy,
    metrics=evaluation_metrics,
    metric_extractor=extract_metrics,
)

scheduler.register("evaluation", every=25_000, callback=evaluator)
```

For each rollout, the evaluator uses `terminal_key` (default `("next", "done")`) to select completed transitions and
passes those complete transitions to `metric_extractor`. The evaluator does not assume where an environment stores
`info`, whether information is replicated across agents, or how agent-specific values should be reduced.

Evaluation temporarily puts the policy in evaluation mode and disables gradient tracking. The previous policy mode is
restored and the evaluation environment is closed even if rollout or metric processing fails. After all episodes,
metrics are flushed using the step supplied directly or by `Scheduler`.

## Configuration and validation

Configuration and envelope dataclasses are frozen, slotted, keyword-only, passive value objects. Validation is explicit:
`validate_collector_config`, `validate_evaluation_config`, `validate_metric_spec`, and `validate_checkpoint` can be called
directly, and the corresponding runtime boundary calls the same validator before using a value. Invalid runtime types
raise `TypeError`; values of the expected type that violate a domain rule raise `ValueError`.

Mutable mappings such as `collector_kwargs`, `rollout_kwargs`, checkpoint state, metadata, and metric labels are accepted
as caller-owned objects. The package does not copy them or promise deep immutability.

## Development

From the repository root, run the complete package suite with:

```bash
uv run pytest packages/rl-core/tests
```
