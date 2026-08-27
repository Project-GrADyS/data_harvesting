from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import mlflow
from mlflow import MlflowClient
import pandas as pd

from data_harvesting.eval import _logged_policy_model_metadata
from data_harvesting.eval import eval as run_evaluation
from data_harvesting.eval import load_policy_from_model_id


_MODEL_COLUMNS = [
    "name",
    "model_id",
    "kind",
    "step",
    "step_inferred",
    "creation_time",
]


def _escape_mlflow_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _parse_logged_config(run_id: str, params: Mapping[str, str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in params.items():
        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed_value = value

        if "." not in key:
            config[key] = parsed_value
            continue

        section = config
        parts = key.split(".")
        for part in parts[:-1]:
            section = section.setdefault(part, {})
        section[parts[-1]] = parsed_value

    if "environment" not in config:
        raise ValueError(
            f"Run '{run_id}' does not include a logged environment configuration."
        )
    return config


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _normalize_metric_names(
    names: Iterable[str] | str | None,
    *,
    available: tuple[str, ...],
) -> tuple[str, ...]:
    if names is None:
        return available
    if isinstance(names, str):
        requested = (names,)
    else:
        requested = tuple(dict.fromkeys(names))

    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(
            f"Unknown metrics: {', '.join(unknown)}. "
            f"Available metrics: {', '.join(available)}"
        )
    return requested


@dataclass(frozen=True)
class RunModel:
    """A model logged by an :class:`ExperimentRun`."""

    run: ExperimentRun = field(repr=False, compare=False)
    name: str
    model_id: str
    kind: str
    step: int | None
    step_inferred: bool
    creation_timestamp: int | None

    @cached_property
    def _loaded_policy(self):
        return load_policy_from_model_id(
            self.model_id,
            tracking_uri=self.run.tracking_uri,
        )

    def load(self):
        """Load and cache the logged PyTorch policy from MLflow."""
        return self._loaded_policy

    def evaluate(
        self,
        num_runs: int,
        *,
        config_overrides: Mapping[str, Any] | None = None,
        seed: int | None = None,
        visual: bool = False,
    ) -> pd.DataFrame:
        """Evaluate this model and return one row per episode."""
        config = self.run.config
        if config_overrides is not None:
            config = _deep_merge(config, config_overrides)

        results = run_evaluation(
            self.load(),
            config,
            num_runs,
            visual=visual,
            seed=seed,
        )
        episodes = pd.DataFrame(results.get("episodes", []))
        provenance = {
            "source_run_id": self.run.run_id,
            "source_run_name": self.run.run_name,
            "model_name": self.name,
            "model_id": self.model_id,
            "model_step": self.step,
            "model_kind": self.kind,
            "model_step_inferred": self.step_inferred,
        }
        for position, (column, value) in enumerate(provenance.items()):
            episodes.insert(position, column, value)
        return episodes


class ExperimentRun:
    """Notebook-friendly access to one MLflow experiment run."""

    def __init__(self, *, run, client: MlflowClient, tracking_uri: str) -> None:
        self._run = run
        self._client = client
        self.tracking_uri = tracking_uri
        self._metric_history_cache: dict[str, tuple[Any, ...]] = {}
        self._models_cache: tuple[RunModel, ...] | None = None
        self._config = _parse_logged_config(run.info.run_id, run.data.params)

    @classmethod
    def from_id(
        cls,
        run_id: str,
        *,
        tracking_uri: str | None = None,
    ) -> ExperimentRun:
        """Reference a run by its MLflow run ID."""
        resolved_uri = tracking_uri or mlflow.get_tracking_uri()
        client = MlflowClient(tracking_uri=resolved_uri)
        return cls(run=client.get_run(run_id), client=client, tracking_uri=resolved_uri)

    @classmethod
    def from_name(
        cls,
        experiment_name: str,
        run_name: str,
        *,
        tracking_uri: str | None = None,
    ) -> ExperimentRun:
        """Reference the uniquely named run in an MLflow experiment."""
        resolved_uri = tracking_uri or mlflow.get_tracking_uri()
        client = MlflowClient(tracking_uri=resolved_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"MLflow experiment '{experiment_name}' was not found.")

        matches = [
            run
            for run in client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=(
                    "tags.`mlflow.runName` = "
                    f"'{_escape_mlflow_filter_value(run_name)}'"
                ),
                max_results=50_000,
            )
            if run.info.run_name == run_name
        ]
        if not matches:
            raise ValueError(
                f"Run '{run_name}' was not found in experiment '{experiment_name}'."
            )
        if len(matches) > 1:
            run_ids = ", ".join(sorted(run.info.run_id for run in matches))
            raise ValueError(
                f"Run name '{run_name}' is not unique in experiment "
                f"'{experiment_name}'. Matching run IDs: {run_ids}"
            )
        return cls(run=matches[0], client=client, tracking_uri=resolved_uri)

    @property
    def run_id(self) -> str:
        return self._run.info.run_id

    @property
    def run_name(self) -> str | None:
        return self._run.info.run_name

    @property
    def experiment_id(self) -> str:
        return self._run.info.experiment_id

    @property
    def status(self) -> str:
        return self._run.info.status

    @property
    def config(self) -> dict[str, Any]:
        return deepcopy(self._config)

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._run.data.metrics))

    def _metric_history(self, name: str) -> tuple[Any, ...]:
        if name not in self._metric_history_cache:
            self._metric_history_cache[name] = tuple(
                self._client.get_metric_history(self.run_id, name)
            )
        return self._metric_history_cache[name]

    def metrics(
        self,
        names: Iterable[str] | str | None = None,
    ) -> pd.DataFrame:
        """Return selected metric histories as a wide table keyed by step."""
        selected = _normalize_metric_names(names, available=self.metric_names)
        records_by_step: dict[int, dict[str, float | int]] = {}
        for metric_name in selected:
            points = sorted(
                self._metric_history(metric_name),
                key=lambda point: (point.step, point.timestamp),
            )
            for point in points:
                record = records_by_step.setdefault(point.step, {"step": point.step})
                record[metric_name] = point.value

        columns = ["step", *selected]
        if not records_by_step:
            return pd.DataFrame(columns=columns)
        return (
            pd.DataFrame(records_by_step.values(), columns=columns)
            .sort_values("step")
            .reset_index(drop=True)
        )

    def _latest_metric_step(self) -> int | None:
        latest_step: int | None = None
        for metric_name in self.metric_names:
            for point in self._metric_history(metric_name):
                latest_step = (
                    point.step if latest_step is None else max(latest_step, point.step)
                )
        return latest_step

    def _models(self) -> tuple[RunModel, ...]:
        if self._models_cache is not None:
            return self._models_cache

        logged_models = list(
            self._client.search_logged_models(
                experiment_ids=[self.experiment_id],
                filter_string=f"source_run_id = '{self.run_id}'",
            )
        )
        final_step = (
            self._latest_metric_step()
            if any(model.name == "policy_model" for model in logged_models)
            else None
        )
        models = []
        for model in logged_models:
            metadata = _logged_policy_model_metadata(
                name=model.name,
                model_id=model.model_id,
                creation_timestamp=model.creation_timestamp,
                final_step=final_step,
            )
            models.append(
                RunModel(
                    run=self,
                    name=metadata.name,
                    model_id=metadata.model_id,
                    kind=metadata.kind,
                    step=metadata.step,
                    step_inferred=metadata.step_inferred,
                    creation_timestamp=metadata.creation_timestamp,
                )
            )
        self._models_cache = tuple(
            sorted(
                models,
                key=lambda model: (
                    model.step is None,
                    model.step if model.step is not None else 0,
                    {"checkpoint": 0, "final": 1}.get(model.kind, 2),
                    model.name,
                    model.model_id,
                ),
            )
        )
        return self._models_cache

    def models(self) -> pd.DataFrame:
        """List models logged by the run and their training timesteps."""
        rows = [
            {
                "name": model.name,
                "model_id": model.model_id,
                "kind": model.kind,
                "step": model.step,
                "step_inferred": model.step_inferred,
                "creation_time": pd.to_datetime(
                    model.creation_timestamp,
                    unit="ms",
                    utc=True,
                )
                if model.creation_timestamp is not None
                else pd.NaT,
            }
            for model in self._models()
        ]
        table = pd.DataFrame(rows, columns=_MODEL_COLUMNS)
        table["step"] = pd.array(table["step"], dtype="Int64")
        return table

    def models_with_metrics(
        self,
        names: Iterable[str] | str | None = None,
    ) -> pd.DataFrame:
        """Join models with metric values logged at exactly the same step."""
        selected = (
            tuple(name for name in self.metric_names if name.startswith("eval/"))
            if names is None
            else _normalize_metric_names(names, available=self.metric_names)
        )
        models = self.models()
        if not selected:
            return models
        metrics = self.metrics(selected)
        metrics["step"] = pd.array(metrics["step"], dtype="Int64")
        return models.merge(metrics, on="step", how="left", sort=False)

    def checkpoint(self, step: int) -> RunModel:
        """Select the unique numbered checkpoint logged at ``step``."""
        matches = [
            model
            for model in self._models()
            if model.kind == "checkpoint" and model.step == step
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            model_ids = ", ".join(model.model_id for model in matches)
            raise ValueError(
                f"Multiple checkpoints were logged at step {step}: {model_ids}"
            )
        available = ", ".join(
            str(model.step)
            for model in self._models()
            if model.kind == "checkpoint"
        )
        raise ValueError(
            f"No checkpoint was logged at step {step}. Available steps: {available or 'none'}"
        )

    def final_model(self) -> RunModel:
        """Select the unique unnumbered final policy model."""
        matches = [model for model in self._models() if model.kind == "final"]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            model_ids = ", ".join(model.model_id for model in matches)
            raise ValueError(f"Multiple final models were logged: {model_ids}")
        raise ValueError(f"Run '{self.run_id}' has no final policy model.")
