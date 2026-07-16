from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "params.yaml"
DEFAULT_TRACKING_URI = f"file:{PROJECT_ROOT / 'mlruns'}"
