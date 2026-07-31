from pathlib import Path

from rl_core import MetricsCollector

from sushigo.config import load_config, with_preset
from sushigo.environment import make_env
from sushigo.metrics import (
    environment_metric_specs,
    extract_batch_metrics,
    extract_terminal_metrics,
)
from sushigo.policy import build_q_policy

PARAMS = Path(__file__).parents[1] / "params.yaml"


def test_metrics_ignore_padding_and_collect_terminal_scores():
    config = with_preset(load_config(PARAMS), "variable_2_4")
    environment = make_env(config)
    policy = build_q_policy(environment, config, device="cpu")
    rollout = environment.rollout(max_steps=40, policy=policy)
    values = extract_batch_metrics(rollout)
    collector = MetricsCollector(specs=environment_metric_specs())
    collector.push(values)
    snapshot = collector.peek()
    assert "mean_turn_reward" in snapshot
    assert "mean_final_score" in snapshot
    assert sum(
        snapshot.get(key, 0)
        for key in ("episodes_2p", "episodes_3p", "episodes_4p")
    ) == 1
    terminal = extract_terminal_metrics(rollout[-1:])
    assert terminal["active_players"].numel() == 1
    environment.close()
