# AGENTS.md

## Purpose

This repository trains and evaluates multi-agent data-harvesting policies with PyTorch, TorchRL, and MLflow. Agents working in this repo should prefer small, targeted changes that preserve the current training and evaluation workflows.

## Repository Layout

- `data_harvesting/`: core package code.
- `data_harvesting/environment/`: environment dynamics, metrics, protocols, and wrappers.
- `data_harvesting/encoder/`: encoder blocks and flex encoder variants.
- `tests/`: unit tests for algorithms, collectors, replay buffer, metrics, environment behavior, and flex encoder paths.
- `main.py`: primary training entrypoint using `params.yaml`.
- `evaluate.py`: evaluate a saved MLflow run by run ID.
- `tune.py` and `tune_worker.py`: hyperparameter tuning entrypoints.
- `profile_training.py`: profiling-oriented training runner.
- `params.yaml`: default runtime configuration.
- `mlruns/`: local MLflow tracking artifacts. Treat as generated output unless the task is explicitly about MLflow results.

## Environment And Tooling

- Python: `>=3.11` as defined in `pyproject.toml`.
- Dependency manager: `uv`.
- Install dependencies with `uv sync`.
- Run Python entrypoints with `uv run ...` so the project environment is used consistently.

## Common Commands

- Run all tests: `uv run pytest`
- Run a focused test file: `uv run pytest tests/test_algorithm_maddpg.py`
- Run training with default params: `uv run python main.py`
- Run training with an experiment name: `uv run python main.py -E <experiment_name>`
- Evaluate a saved run: `uv run python evaluate.py --run-id <MLFLOW_RUN_ID> --num-runs <N>`
- Run the profiler entrypoint: `uv run python profile_training.py`
- Run tuning: `uv run python tune.py`

## Working Rules

- Keep changes scoped to the user request. Do not refactor unrelated training or environment code opportunistically.
- Avoid editing `mlruns/`, `.venv/`, `venv/`, `__pycache__/`, or `.pytest_cache/` unless the task explicitly requires it.
- New tests should go under `tests/` near the behavior they cover.
- If changing environment observations, rewards, metrics, or masking, run the relevant `tests/environment/` coverage.
- If changing encoder code, run the matching `tests/flex_encoder/` coverage.
- If changing training, optimization, replay, collector, or algorithm code, run the affected top-level tests in `tests/`.
- Prefer CPU-safe tests and short feedback loops. Do not introduce heavyweight training runs as part of validation unless the task requires it.

## MLflow Notes

- The project uses a local MLflow tracking URI at `file:./mlruns` in the main scripts.
- Training and evaluation workflows depend on MLflow run IDs and logged model artifacts.
- Do not delete or rewrite existing MLflow artifacts unless the user explicitly asks for cleanup or migration work.

## Coding Conventions

- Match the existing style in surrounding files. The codebase currently uses straightforward typed Python without enforcing a formatter in repo config.
- Prefer targeted helper functions and explicit config-driven behavior over hardcoded constants.
- Preserve existing public CLI arguments and config keys unless the task explicitly calls for interface changes.

## Validation Expectations

- At minimum, run the most relevant tests for the files you changed.
- If you cannot run validation, state that clearly in the final handoff.
- If a change affects CLI behavior or config loading, verify the relevant entrypoint help or execution path when practical.
