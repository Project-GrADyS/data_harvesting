from pathlib import Path

from sushigo.algorithm import DQNAlgorithm
from sushigo.config import load_config, with_preset
from sushigo.environment import make_env

PARAMS = Path(__file__).parents[1] / "params.yaml"


def test_dqn_learning_step_is_cpu_safe():
    config = with_preset(load_config(PARAMS), "fixed_2p")
    config["replay_buffer"]["batch_size"] = 8
    config["replay_buffer"]["capacity"] = 100
    config["optimization"]["updates_per_batch"] = 1
    environment = make_env(config)
    algorithm = DQNAlgorithm(environment, config, device=environment.device)
    rollout = environment.rollout(
        max_steps=30, policy=algorithm.exploratory_policy
    )
    metrics = algorithm.learn(rollout)
    assert float(metrics["loss"]) >= 0
    assert 0 <= float(metrics["epsilon"]) <= 1
    environment.close()
