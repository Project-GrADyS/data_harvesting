from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mlflow
import yaml

from data_harvesting.eval import eval as run_eval
from data_harvesting.eval import LoggedPolicyModel
from data_harvesting.eval import load_config_from_mlflow_run
from data_harvesting.eval import load_policy_from_model_id
from data_harvesting.eval import load_policy_from_mlflow_run
from data_harvesting.eval import list_policy_models_from_mlflow_run
from _paths import DEFAULT_TRACKING_URI


def _print_summary(results: dict) -> None:
    print("\n=== Evaluation Summary ===")
    print(f"Runs: {results['num_runs']}")

    print("\nMetrics:")
    for metric_name, values in results["metrics"].items():
        print(
            f"- {metric_name}: "
            f"mean={values['mean']:.4f}, "
            f"std={values['std']:.4f}, "
            f"min={values['min']:.4f}, "
            f"max={values['max']:.4f}"
        )

    print("\nEnd causes:")
    counts = results["end_cause_counts"]
    rates = results["end_cause_rate"]
    for cause_name in counts:
        print(f"- {cause_name}: count={counts[cause_name]}, rate={rates[cause_name]:.2%}")

    scenario_metrics = results.get("scenario_metrics", {})
    if scenario_metrics:
        print("\nScenario breakdown:")
        for scenario_key, scenario_results in sorted(scenario_metrics.items()):
            scenario = scenario_results["scenario"]
            print(
                f"\n[{scenario_key}] "
                f"agents={scenario['agents']}, sensors={scenario['sensors']}, runs={scenario_results['num_runs']}"
            )
            for metric_name, values in scenario_results["metrics"].items():
                print(
                    f"- {metric_name}: "
                    f"mean={values['mean']:.4f}, "
                    f"std={values['std']:.4f}, "
                    f"min={values['min']:.4f}, "
                    f"max={values['max']:.4f}"
                )
            counts = scenario_results["end_cause_counts"]
            rates = scenario_results["end_cause_rate"]
            for cause_name in counts:
                print(
                    f"- end_cause[{cause_name}]: "
                    f"count={counts[cause_name]}, rate={rates[cause_name]:.2%}"
                )


def _write_output_table(results: dict, output_path: str) -> None:
    episode_rows = results.get("episodes", [])
    scenario_metrics = results.get("scenario_metrics", {})

    metric_names = list(results.get("metrics", {}).keys())
    end_cause_names = list(results.get("end_cause_counts", {}).keys())
    if episode_rows:
        identity_columns = [
            column
            for column in ("model_name", "model_id")
            if any(column in row for row in episode_rows)
        ]
        categorical_metric_keys = sorted(
            {
                key
                for row in episode_rows
                for key in row.keys()
                if key
                not in {
                    *identity_columns,
                    "run_index",
                    "scenario_key",
                    "num_agents",
                    "num_sensors",
                    *metric_names,
                }
            }
        )
        columns = [
            *identity_columns,
            "run_index",
            "scenario_key",
            "num_agents",
            "num_sensors",
            *metric_names,
            *categorical_metric_keys,
        ]
    else:
        columns = ["scenario_key", "agents", "sensors", "num_runs"]
        for metric_name in metric_names:
            columns.extend(
                [
                    f"{metric_name}_mean",
                    f"{metric_name}_std",
                    f"{metric_name}_min",
                    f"{metric_name}_max",
                ]
            )
        for cause_name in end_cause_names:
            columns.append(f"end_cause_{cause_name}_count")
            columns.append(f"end_cause_{cause_name}_rate")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        if episode_rows:
            for episode_row in episode_rows:
                writer.writerow({col: episode_row.get(col, "") for col in columns})
        elif scenario_metrics:
            for scenario_key, scenario_results in sorted(scenario_metrics.items()):
                row = {
                    "scenario_key": scenario_key,
                    "agents": scenario_results["scenario"]["agents"],
                    "sensors": scenario_results["scenario"]["sensors"],
                    "num_runs": scenario_results["num_runs"],
                }

                for metric_name in metric_names:
                    values = scenario_results["metrics"].get(metric_name, {})
                    row[f"{metric_name}_mean"] = values.get("mean", "")
                    row[f"{metric_name}_std"] = values.get("std", "")
                    row[f"{metric_name}_min"] = values.get("min", "")
                    row[f"{metric_name}_max"] = values.get("max", "")

                counts = scenario_results.get("end_cause_counts", {})
                rates = scenario_results.get("end_cause_rate", {})
                for cause_name in end_cause_names:
                    row[f"end_cause_{cause_name}_count"] = counts.get(cause_name, 0)
                    row[f"end_cause_{cause_name}_rate"] = rates.get(cause_name, 0.0)

                writer.writerow(row)
        else:
            row = {
                "scenario_key": "overall",
                "agents": "",
                "sensors": "",
                "num_runs": results.get("num_runs", 0),
            }

            for metric_name in metric_names:
                values = results.get("metrics", {}).get(metric_name, {})
                row[f"{metric_name}_mean"] = values.get("mean", "")
                row[f"{metric_name}_std"] = values.get("std", "")
                row[f"{metric_name}_min"] = values.get("min", "")
                row[f"{metric_name}_max"] = values.get("max", "")

            counts = results.get("end_cause_counts", {})
            rates = results.get("end_cause_rate", {})
            for cause_name in end_cause_names:
                row[f"end_cause_{cause_name}_count"] = counts.get(cause_name, 0)
                row[f"end_cause_{cause_name}_rate"] = rates.get(cause_name, 0.0)

            writer.writerow(row)

    print(f"Wrote CSV table to {output_file}")


def _tag_results_with_model(results: dict, model: LoggedPolicyModel) -> None:
    results["model_name"] = model.name
    results["model_id"] = model.model_id
    for episode_row in results.get("episodes", []):
        episode_row["model_name"] = model.name
        episode_row["model_id"] = model.model_id


def _combine_model_results(model_results: list[dict]) -> dict:
    metric_names = {
        metric_name
        for results in model_results
        for metric_name in results.get("metrics", {})
    }
    return {
        "num_runs": sum(results.get("num_runs", 0) for results in model_results),
        "metrics": {metric_name: {} for metric_name in sorted(metric_names)},
        "episodes": [
            episode_row
            for results in model_results
            for episode_row in results.get("episodes", [])
        ],
    }


def _print_model_comparison(model_results: list[dict]) -> None:
    if len(model_results) <= 1:
        return

    print("\n=== Model Comparison ===")
    best: tuple[float, str] | None = None
    for results in model_results:
        model_name = results.get("model_name", "")
        all_collected_total = sum(
            float(row.get("all_collected", 0.0))
            for row in results.get("episodes", [])
        )
        all_collected_mean = (
            results.get("metrics", {}).get("all_collected", {}).get("mean", 0.0)
        )
        print(
            f"- {model_name}: "
            f"all_collected_total={all_collected_total:.0f}, "
            f"all_collected_mean={all_collected_mean:.4f}"
        )
        if best is None or all_collected_total > best[0]:
            best = (all_collected_total, model_name)

    if best is not None:
        print(f"Best by total all_collected: {best[1]} ({best[0]:.0f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved MLflow model run.")
    parser.add_argument("--run-id", "-R", required=True, help="MLflow run ID to evaluate")
    parser.add_argument(
        "--num-runs",
        "-N",
        required=True,
        type=int,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Run environment in visual mode during evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel evaluation workers (default: 1)",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="Optional YAML params file used to build the environment; defaults to the run's logged config",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--model-name",
        default="policy_model",
        help="Preferred logged model name for the given run",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Evaluate every logged model from the run",
    )
    parser.add_argument(
        "--output-table",
        default=None,
        help="Optional CSV file path to write a per-episode results table",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    if args.params:
        with open(args.params, "r") as handle:
            config: dict = yaml.safe_load(handle)
        config_source = args.params
    else:
        config = load_config_from_mlflow_run(
            args.run_id,
            tracking_uri=args.tracking_uri,
        )
        config_source = "MLflow run params"

    print(f"Evaluating run_id={args.run_id}")
    print(f"Config source={config_source}")
    print(f"Visual mode={'on' if args.visual else 'off'}")
    print(f"Evaluation workers={args.num_workers}")

    if args.all_models:
        models = list_policy_models_from_mlflow_run(
            args.run_id,
            tracking_uri=args.tracking_uri,
        )
    else:
        policy, model_id = load_policy_from_mlflow_run(
            args.run_id,
            tracking_uri=args.tracking_uri,
            model_name=args.model_name,
        )
        models = [LoggedPolicyModel(name=args.model_name, model_id=model_id)]

    model_results = []
    for model_index, model in enumerate(models, start=1):
        print(
            f"\nEvaluating model {model_index}/{len(models)}: "
            f"{model.name} ({model.model_id})"
        )
        if args.all_models:
            policy = load_policy_from_model_id(model.model_id)
        results = run_eval(
            policy,
            config,
            args.num_runs,
            visual=args.visual,
            num_workers=args.num_workers,
        )
        _tag_results_with_model(results, model)
        _print_summary(results)
        model_results.append(results)

    _print_model_comparison(model_results)
    if args.output_table:
        _write_output_table(_combine_model_results(model_results), args.output_table)


if __name__ == "__main__":
    main()
