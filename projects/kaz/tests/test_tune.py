from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import yaml


def _load_tune_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("kaz_tune_script", scripts / "tune.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_one_trial_tuning_writes_decoded_parameters(
    monkeypatch, tmp_path
) -> None:
    tune = _load_tune_module(monkeypatch)
    output = tmp_path / "best_params.yaml"
    tracking = (tmp_path / "mlruns").resolve().as_uri()

    def fake_run(command, **kwargs):
        result_path = Path(command[command.index("--result-path") + 1])
        result_path.write_text(json.dumps({"score": 2.0}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tune.subprocess, "run", fake_run)
    tune.main(
        [
            "--trials",
            "1",
            "--timesteps",
            "16",
            "--tracking-uri",
            tracking,
            "--output",
            str(output),
        ]
    )

    best = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert best["training"]["batch_size"] in {128, 256, 512}
    assert best["training"]["exploration_epsilon_end"] in {0.02, 0.05, 0.1}
    assert best["optimization"]["updates_per_batch"] in {1, 2, 4}
