from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

from data_harvesting.eval import LoggedPolicyModel


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate.py"
SCRIPT_DIRECTORY = str(SCRIPT_PATH.parent)
if SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPT_DIRECTORY)

SPEC = importlib.util.spec_from_file_location("data_harvesting_evaluate_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluate_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_script)


def _model_results(name: str, model_id: str, values: list[float]) -> dict:
    return {
        "num_runs": len(values),
        "metrics": {
            "all_collected": {
                "mean": sum(values) / len(values),
            }
        },
        "episodes": [
            {
                "run_index": index,
                "scenario_key": "agents_1__sensors_1",
                "num_agents": 1,
                "num_sensors": 1,
                "all_collected": value,
            }
            for index, value in enumerate(values)
        ],
        "model_name": name,
        "model_id": model_id,
    }


def test_tag_combine_and_write_multi_model_results(tmp_path) -> None:
    first = _model_results("first", "model-1", [0.0, 1.0])
    second = _model_results("second", "model-2", [1.0, 1.0])
    evaluate_script._tag_results_with_model(
        first,
        LoggedPolicyModel(name="first", model_id="model-1"),
    )
    evaluate_script._tag_results_with_model(
        second,
        LoggedPolicyModel(name="second", model_id="model-2"),
    )

    combined = evaluate_script._combine_model_results([first, second])
    output_path = tmp_path / "evaluation.csv"
    evaluate_script._write_output_table(combined, str(output_path))

    with output_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert combined["num_runs"] == 4
    assert len(rows) == 4
    assert list(rows[0])[:2] == ["model_name", "model_id"]
    assert {(row["model_name"], row["model_id"]) for row in rows} == {
        ("first", "model-1"),
        ("second", "model-2"),
    }


def test_model_comparison_reports_best_total(capsys) -> None:
    first = _model_results("first", "model-1", [0.0, 1.0])
    second = _model_results("second", "model-2", [1.0, 1.0])

    evaluate_script._print_model_comparison([first, second])

    output = capsys.readouterr().out
    assert "first: all_collected_total=1" in output
    assert "second: all_collected_total=2" in output
    assert "Best by total all_collected: second (2)" in output
