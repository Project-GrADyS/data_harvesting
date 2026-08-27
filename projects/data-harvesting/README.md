# Data Harvesting

The data-harvesting project trains and evaluates fixed-slot multi-agent policies
with GrADyS, PyTorch, TorchRL, `flex-marl`, and `rl-core`.

## Layout

```text
projects/data-harvesting/
├── src/data_harvesting/  # Environment, algorithms, and project integration
├── scripts/              # Training, evaluation, tuning, profiling, visualization
├── tests/                # Project-owned tests
├── mlflow_server/        # Private MLflow deployment and backup support
└── params.yaml           # Default runtime configuration
```

Sequential observations expose explicit Boolean `sensors_mask` and
`drones_mask` tensors. `True` means that the corresponding element is valid;
consumers must not infer validity from the `-1` padding value.

## Commands

Run commands from the workspace root:

```bash
uv run --package data-harvesting python projects/data-harvesting/scripts/main.py -E default
uv run --package data-harvesting python projects/data-harvesting/scripts/evaluate.py --run-id <RUN_ID> --num-runs 10
uv run --package data-harvesting python projects/data-harvesting/scripts/tune.py -E tuning
uv run --package data-harvesting python projects/data-harvesting/scripts/profile_training.py
uv run --package data-harvesting python projects/data-harvesting/scripts/visualize_env.py
uv run --package data-harvesting pytest projects/data-harvesting/tests
```

Evaluation uses the configuration logged on the MLflow run by default. Pass
`--params <PATH>` to override it with a local YAML file. To evaluate every
model logged to a run and write a combined per-episode table, run:

```bash
uv run --package data-harvesting python projects/data-harvesting/scripts/evaluate.py \
  --run-id <RUN_ID> --num-runs <N> --all-models --output-table results.csv
```

The combined table includes `model_name` and `model_id` columns. The printed
comparison ranks models by total `all_collected`; evaluation forces
`end_when_all_collected` on so collection success terminates the episode.

Scripts that use a default configuration resolve `params.yaml` relative to this
project. The default MLflow tracking location is
`projects/data-harvesting/mlruns`; set
`MLFLOW_TRACKING_URI` or pass `--tracking-uri` to use the private server.

Older MLflow policies saved as pickled full modules before the `flex-marl`
migration are not compatible with this project layout.

## Notebook analysis

`data_harvesting.analysis` provides a pandas-oriented interface for inspecting
training runs and evaluating their logged policies:

```python
from data_harvesting.analysis import ExperimentRun

run = ExperimentRun.from_name(
    "death",
    "6_mlp",
    tracking_uri="http://localhost:5000",
)

run.metric_names
metrics = run.metrics(["eval/all_collected", "eval/avg_reward"])
models = run.models_with_metrics()

checkpoint = run.checkpoint(1_000_448)
policy = checkpoint.load()
episodes = checkpoint.evaluate(
    100,
    config_overrides={
        "environment": {
            "min_num_agents": 4,
            "max_num_agents": 4,
        }
    },
    seed=1000,
)
```

Run names must be unique within their experiment. Use
`ExperimentRun.from_id(...)` when names are duplicated. Model/metric tables use
exact MLflow steps; the final `policy_model` is assigned the run's maximum
logged metric step and marked with `step_inferred=True`.

A clean starter notebook is available at
`analysis/mlflow_analysis.ipynb`.
