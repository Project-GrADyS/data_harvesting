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

`push()` also accepts a `TensorDictBase` whose top-level keys match configured metric keys. TensorDict support is an
optional dependency and can be installed with `rl-core[tensordict]`.

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
