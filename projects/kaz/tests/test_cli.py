from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_training_cli_help() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--tracking-uri" in result.stdout


def test_tuning_cli_help() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "tune.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--trials" in result.stdout
    assert "--output" in result.stdout
