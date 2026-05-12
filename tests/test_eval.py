import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from data_harvesting.eval import eval as run_eval
from data_harvesting.eval import list_policy_models_from_mlflow_run
from data_harvesting.eval import load_config_from_mlflow_run


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
        "environment": {
            "sequential_obs": True,
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


def test_eval_applies_environment_evaluation_overrides(monkeypatch) -> None:
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
    monkeypatch.setattr(
        "data_harvesting.eval.make_metrics_spec",
        lambda: type("MetricsSpec", (), {"scalar_metrics": [], "categorical_metrics": [], "metrics": []})(),
    )

    source_config = _eval_config()
    run_eval(_make_policy(), source_config, num_runs=1)

    assert captured_configs[0]["environment"]["end_when_all_collected"] is True
    assert source_config["environment"]["end_when_all_collected"] is False


def test_load_config_from_mlflow_run_parses_logged_section_params(monkeypatch) -> None:
    class _FakeClient:
        def get_run(self, run_id: str):
            assert run_id == "run-1"
            return type(
                "Run",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "params": {
                                "environment": "{'max_episode_length': 3, 'sequential_obs': True}",
                                "training": "{'algorithm': 'maddpg'}",
                            }
                        },
                    )()
                },
            )()

    monkeypatch.setattr("data_harvesting.eval.MlflowClient", _FakeClient)

    config = load_config_from_mlflow_run("run-1")

    assert config["environment"] == {"max_episode_length": 3, "sequential_obs": True}
    assert config["training"] == {"algorithm": "maddpg"}


def test_load_config_from_mlflow_run_supports_dotted_params(monkeypatch) -> None:
    class _FakeClient:
        def get_run(self, run_id: str):
            return type(
                "Run",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "params": {
                                "environment.max_episode_length": "3",
                                "environment.sequential_obs": "True",
                            }
                        },
                    )()
                },
            )()

    monkeypatch.setattr("data_harvesting.eval.MlflowClient", _FakeClient)

    config = load_config_from_mlflow_run("run-1")

    assert config["environment"] == {"max_episode_length": 3, "sequential_obs": True}


def test_list_policy_models_from_mlflow_run_returns_all_models(monkeypatch) -> None:
    def _model(name: str, model_id: str, creation_timestamp: int):
        return type(
            "LoggedModel",
            (),
            {
                "name": name,
                "model_id": model_id,
                "creation_timestamp": creation_timestamp,
            },
        )()

    class _FakeClient:
        def get_run(self, run_id: str):
            assert run_id == "run-1"
            return type(
                "Run",
                (),
                {
                    "info": type(
                        "Info",
                        (),
                        {"experiment_id": "experiment-1"},
                    )()
                },
            )()

        def search_logged_models(self, *, experiment_ids: list[str], filter_string: str):
            assert experiment_ids == ["experiment-1"]
            assert filter_string == "source_run_id = 'run-1'"
            return [
                _model("auxiliary_model", "auxiliary", 1),
                _model("policy_checkpoint_step_200", "checkpoint-200", 2),
                _model("policy_model", "final", 4),
                _model("policy_checkpoint_step_100", "checkpoint-100", 3),
            ]

    monkeypatch.setattr("data_harvesting.eval.MlflowClient", _FakeClient)

    models = list_policy_models_from_mlflow_run("run-1")

    assert [model.name for model in models] == [
        "auxiliary_model",
        "policy_checkpoint_step_200",
        "policy_checkpoint_step_100",
        "policy_model",
    ]
    assert [model.model_id for model in models] == [
        "auxiliary",
        "checkpoint-200",
        "checkpoint-100",
        "final",
    ]
