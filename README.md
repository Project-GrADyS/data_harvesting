# Data Harvesting

## MLflow tracking

Local file-backed MLflow runs still work by default, but the preferred setup for
new runs is the private server in `mlflow_server/`.

```bash
export MLFLOW_TRACKING_URI=http://<vpn-host>:5000
uv run python main.py -E default
```

You can also pass the URI per command:

```bash
uv run python main.py --tracking-uri http://<vpn-host>:5000 -E default
uv run python tune.py --tracking-uri http://<vpn-host>:5000 -E tuning
```

## Evaluate a trained run

Use `evaluate.py` to load a saved model from an MLflow **run ID**, execute it in eval mode for `N` runs, and print a terminal summary of metrics.

### Basic usage

```bash
python evaluate.py --run-id <MLFLOW_RUN_ID> --num-runs <N>
```

Example:

```bash
python evaluate.py --run-id 3d577e1a165842ccaa6d0ecb34c2dd35 --num-runs 10
```

### Visual mode

To run evaluation with environment visualization enabled:

```bash
python evaluate.py --run-id <MLFLOW_RUN_ID> --num-runs 1 --visual
```

### Arguments

- `--run-id`, `-R` (required): MLflow run ID to evaluate.
- `--num-runs`, `-N` (required): number of evaluation episodes.
- `--visual`: enable environment visual mode.
- `--params`: path to params YAML (default: `params.yaml`).
- `--tracking-uri`: MLflow tracking URI (default: `file:./mlruns`).
- `--model-name`: preferred logged model name (default: `policy_model`).

### Notes

- The evaluator resolves the saved model from the given MLflow run and runs the policy in eval mode.
- Output includes aggregated statistics for environment metrics (mean/std/min/max) and end-cause counts/rates.
