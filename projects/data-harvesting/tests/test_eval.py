from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from data_harvesting.eval import eval as run_eval
from data_harvesting.eval import list_policy_models_from_mlflow_run
from data_harvesting.eval import load_config_from_mlflow_run
from data_harvesting.eval import load_policy_from_model_id


class ConstantDirectionPolicy(nn.Module):
    def __init__(self, direction: float = 0.0, speed: float = 0.0):
        super().__init__()
        self.direction = float(direction)
        self.speed = float(speed)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        action = torch.zeros(mask.shape + (2,), dtype=torch.float32, device=mask.device)
        action[..., 0] = self.direction
        action[..., 1] = self.speed
        return action


def _make_policy() -> TensorDictModule:
    return TensorDictModule(
        module=ConstantDirectionPolicy(direction=0.0, speed=0.0),
        in_keys=[("agents", "mask")],
        out_keys=[("agents", "action")],
    )


def _eval_config() -> dict:
    return {
        "flex_encoder": {"enabled": True},
        "environment": {
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 1,
            "max_num_agents": 1,
            "min_num_sensors": 1,
            "max_num_sensors": 1,
            "scenario_size": 10.0,
            "max_episode_length": 3,
            "max_seconds_stalled": 1,
            "communication_range": 0.0,
            "state_num_closest_sensors": 1,
            "state_num_closest_drones": 1,
            "id_on_state": True,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": False,
        }
    }


def test_eval_summarizes_dynamic_scalar_and_categorical_metrics() -> None:
    results = run_eval(_make_policy(), _eval_config(), num_runs=3, seed=100)

    assert results["num_runs"] == 3
    assert "avg_reward" in results["metrics"]
    assert "completion_time" in results["metrics"]
    assert results["metrics"]["avg_reward"]["mean"] == pytest.approx(-1.0)
    assert results["metrics"]["avg_reward"]["std"] == pytest.approx(0.0)
    assert results["metrics"]["episode_duration"]["mean"] == pytest.approx(2.0)
    assert results["metrics"]["completion_time"]["mean"] == pytest.approx(3.0)
    assert results["end_cause_counts"]["STALLED"] == 3
    assert results["end_cause_rate"]["STALLED"] == pytest.approx(1.0)
    assert results["end_cause_counts"]["ALL_COLLECTED"] == 0
    assert "scenario_metrics" in results

    scenario_results = results["scenario_metrics"]["agents_1__sensors_1"]
    assert scenario_results["scenario"] == {"agents": 1, "sensors": 1}
    assert scenario_results["num_runs"] == 3
    assert scenario_results["metrics"]["avg_reward"]["mean"] == pytest.approx(-1.0)
    assert scenario_results["metrics"]["completion_time"]["mean"] == pytest.approx(3.0)
    assert scenario_results["end_cause_counts"]["STALLED"] == 3
    assert scenario_results["end_cause_rate"]["STALLED"] == pytest.approx(1.0)


def test_eval_applies_environment_overrides_without_mutating_source(monkeypatch) -> None:
    captured_configs: list[dict] = []

    class _FakeRollout:
        def get(self, key):
            assert key == ("next", "agents", "info")
            return TensorDict(
                {
                    "num_agents": torch.tensor([[1.0]]),
                    "num_sensors": torch.tensor([[1.0]]),
                },
                batch_size=[1, 1],
            )

    class _FakeEnv:
        def rollout(self, *, max_steps: int, policy: nn.Module):
            return _FakeRollout()

        def close(self) -> None:
            return None

    def _make_env(config: dict) -> _FakeEnv:
        captured_configs.append(config)
        return _FakeEnv()

    monkeypatch.setattr("data_harvesting.eval.make_env", _make_env)
    monkeypatch.setattr("data_harvesting.eval.make_metrics_spec", lambda: ())

    source_config = _eval_config()
    run_eval(_make_policy(), source_config, num_runs=1)

    assert captured_configs[0]["environment"]["end_when_all_collected"] is True
    assert source_config["environment"]["end_when_all_collected"] is False


def test_load_config_from_mlflow_run_parses_nested_and_dotted_params(monkeypatch) -> None:
    class _FakeClient:
        def get_run(self, run_id: str):
            assert run_id == "run-123"
            return SimpleNamespace(
                data=SimpleNamespace(
                    params={
                        "environment": "{'max_episode_length': 50, 'sequential_obs': True}",
                        "training.algorithm": "maddpg",
                        "evaluation.seed": "None",
                        "label": "experiment-a",
                    }
                )
            )

    monkeypatch.setattr("data_harvesting.eval.MlflowClient", _FakeClient)

    config = load_config_from_mlflow_run("run-123")

    assert config == {
        "environment": {"max_episode_length": 50, "sequential_obs": True},
        "training": {"algorithm": "maddpg"},
        "evaluation": {"seed": None},
        "label": "experiment-a",
    }


def test_load_config_from_mlflow_run_requires_environment_config(monkeypatch) -> None:
    class _FakeClient:
        def get_run(self, run_id: str):
            return SimpleNamespace(data=SimpleNamespace(params={"training.lr": "0.001"}))

    monkeypatch.setattr("data_harvesting.eval.MlflowClient", _FakeClient)

    with pytest.raises(ValueError, match="pass --params"):
        load_config_from_mlflow_run("missing-environment")


def test_list_policy_models_from_mlflow_run_returns_deterministic_order(monkeypatch) -> None:
    logged_models = [
        SimpleNamespace(name="policy_model", model_id="final", creation_timestamp=300),
        SimpleNamespace(name="checkpoint", model_id="checkpoint-b", creation_timestamp=100),
        SimpleNamespace(name="checkpoint", model_id="checkpoint-a", creation_timestamp=100),
        SimpleNamespace(name="auxiliary", model_id="aux", creation_timestamp=None),
    ]

    class _FakeClient:
        def get_run(self, run_id: str):
            assert run_id == "run-123"
            return SimpleNamespace(info=SimpleNamespace(experiment_id="experiment-1"))

        def search_logged_models(self, **kwargs):
            assert kwargs == {
                "experiment_ids": ["experiment-1"],
                "filter_string": "source_run_id = 'run-123'",
            }
            return logged_models

    monkeypatch.setattr("data_harvesting.eval.MlflowClient", _FakeClient)

    models = list_policy_models_from_mlflow_run("run-123")

    assert [model.model_id for model in models] == [
        "aux",
        "checkpoint-a",
        "checkpoint-b",
        "final",
    ]


def test_list_policy_models_from_mlflow_run_rejects_empty_run(monkeypatch) -> None:
    class _FakeClient:
        def get_run(self, run_id: str):
            return SimpleNamespace(info=SimpleNamespace(experiment_id="experiment-1"))

        def search_logged_models(self, **kwargs):
            return []

    monkeypatch.setattr("data_harvesting.eval.MlflowClient", _FakeClient)

    with pytest.raises(ValueError, match="No logged models"):
        list_policy_models_from_mlflow_run("empty-run")


def test_load_policy_from_model_id_uses_mlflow_model_uri(monkeypatch) -> None:
    loaded_uris: list[str] = []
    expected_policy = object()
    monkeypatch.setattr(
        "data_harvesting.eval.mlflow_pytorch.load_model",
        lambda uri: loaded_uris.append(uri) or expected_policy,
    )

    policy = load_policy_from_model_id("model-123")

    assert policy is expected_policy
    assert loaded_uris == ["models:/model-123"]
