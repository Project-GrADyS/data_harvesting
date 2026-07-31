from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "params.yaml"
DEFAULT_TRACKING_URI = (PROJECT_ROOT / "mlruns").resolve().as_uri()
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
