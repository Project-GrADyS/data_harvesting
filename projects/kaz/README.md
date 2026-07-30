# KAZ MADDPG

This project is a deliberately small MADDPG training loop for PettingZoo's
Knights Archers Zombies (`knights_archers_zombies_v10`) environment.

It uses KAZ's experimental variable-length vector observations, pads them at
the environment boundary, and passes the entity values and validity masks to
the `flex-marl` Transformer encoder. All agents share one role-aware actor. A
centralized critic learns a cooperative team value from all observations and
one-hot actions.

KAZ has discrete actions, so the actor uses hard one-hot actions with a
straight-through softmax gradient estimator. Collection adds annealed
epsilon-greedy exploration; evaluation always uses the actor's deterministic
argmax action.

## Training

```powershell
uv run --package kaz-training python projects/kaz/scripts/train.py `
  --experiment kaz-maddpg `
  --run-name baseline
```

Configuration defaults live in `projects/kaz/params.yaml`. Override the
configuration file, MLflow location, seed, or training device with:

```powershell
uv run --package kaz-training python projects/kaz/scripts/train.py `
  --params projects/kaz/params.yaml `
  --tracking-uri file:///D:/mlruns `
  --seed 7 `
  --device auto
```

By default, MLflow runs and policy artifacts are written to
`projects/kaz/mlruns`.

## Hyperparameter tuning

```powershell
uv run --package kaz-training python projects/kaz/scripts/tune.py `
  --trials 20 `
  --timesteps 100000 `
  --output projects/kaz/best_params.yaml
```

Each Hyperopt trial runs in a fresh process, logs to MLflow, and maximizes
deterministic evaluation team reward (the total number of zombie kills).

## Tests

```powershell
uv run --package kaz-training pytest projects/kaz/tests
```

The tests use SDL's headless video driver and short episodes; they do not run a
full training job.
