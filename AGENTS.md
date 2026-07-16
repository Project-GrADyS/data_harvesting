# AGENTS.md

## Purpose

This repository trains and evaluates multi-agent data-harvesting policies with PyTorch, TorchRL, and MLflow. Agents working in this repo should prefer small, targeted changes that preserve the current training and evaluation workflows.

## Repository Layout

- `packages/`: reusable `flex-marl`, `rl-core`, and `validation-core` packages.
- `projects/data-harvesting/src/data_harvesting/`: project package code.
- `projects/data-harvesting/src/data_harvesting/environment/`: environment dynamics, metrics, protocols, and wrappers.
- `projects/data-harvesting/scripts/`: training, evaluation, tuning, profiling, and visualization entrypoints.
- `projects/data-harvesting/tests/`: project-owned tests.
- `projects/data-harvesting/params.yaml`: default runtime configuration.
- `projects/data-harvesting/mlflow_server/`: private MLflow deployment files.
- `mlruns/`: local MLflow tracking artifacts. Treat as generated output unless the task is explicitly about MLflow results.

## Environment And Tooling

- Python: `>=3.11` as defined in `pyproject.toml`.
- Dependency manager: `uv`.
- Install dependencies with `uv sync`.
- Run Python entrypoints with `uv run ...` so the project environment is used consistently.

## Common Commands

- Run all tests: `uv run pytest packages projects/data-harvesting/tests`
- Run project tests: `uv run --package data-harvesting pytest projects/data-harvesting/tests`
- Run a focused test: `uv run --package data-harvesting pytest projects/data-harvesting/tests/test_algorithm_maddpg.py`
- Run training: `uv run --package data-harvesting python projects/data-harvesting/scripts/main.py`
- Evaluate a run: `uv run --package data-harvesting python projects/data-harvesting/scripts/evaluate.py --run-id <MLFLOW_RUN_ID> --num-runs <N>`
- Run profiling: `uv run --package data-harvesting python projects/data-harvesting/scripts/profile_training.py`
- Run tuning: `uv run --package data-harvesting python projects/data-harvesting/scripts/tune.py`

## Working Rules

- Keep changes scoped to the user request. Do not refactor unrelated training or environment code opportunistically.
- Avoid editing `mlruns/`, `.venv/`, `venv/`, `__pycache__/`, or `.pytest_cache/` unless the task explicitly requires it.
- New project tests should go under `projects/data-harvesting/tests/` near the behavior they cover.
- New reusable-package tests should stay with their package under `packages/*/tests/`.
- If changing environment observations, rewards, metrics, or masking, run the project `tests/environment/` coverage.
- If changing encoder integration, run the project `tests/flex_encoder/` and `packages/flex-marl/tests/` coverage.
- Prefer CPU-safe tests and short feedback loops. Do not introduce heavyweight training runs as part of validation unless the task requires it.

## MLflow Notes

- The scripts default to `projects/data-harvesting/mlruns` for local tracking.
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
