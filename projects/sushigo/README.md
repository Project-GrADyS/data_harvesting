# Sushi Go

This uv workspace project trains a shared independent DQN policy for the
simultaneous-move Sushi Go environment. It supports fixed two-, three-, or
four-player tables and a single variable-player policy with four dense slots.

The structured encoder uses the workspace `flex-marl` package. Collection,
metrics, scheduling, and evaluation use `rl-core`. MLflow is the authoritative
store for parameters, metrics, checkpoints, and league results.

## Commands

From the repository root:

```powershell
uv run --package sushigo python projects/sushigo/scripts/train.py
uv run --package sushigo python projects/sushigo/scripts/train_league.py --dry-run
uv run --package sushigo python projects/sushigo/scripts/run_league.py --players 2
uv run --package sushigo pytest projects/sushigo/tests
```

Training reads `params.yaml`. `train_league.py` launches fixed-player, variable
MLP, and variable encoder presets with independently tagged repetitions.
`run_league.py` discovers all compatible finished runs from MLflow, excludes two
repetitions of the same family from a table, adds a legal-action random baseline,
and logs both aggregate standings and per-game CSV data back to MLflow.

The default tracking store is the ignored `projects/sushigo/mlruns` directory.
Pass `--tracking-uri` and `--experiment` to any script to use another MLflow
deployment.

## Checkpoints

Checkpoints are versioned dictionaries logged at `policy/checkpoint.pt`. They
contain the policy state, model configuration, and environment configuration.
Artifacts from the pre-workspace Sushi Go project are intentionally unsupported.
