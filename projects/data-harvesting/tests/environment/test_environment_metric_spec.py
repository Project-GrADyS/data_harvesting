from rl_core import CategoricalMetricSpec, ScalarMetricSpec, ScalarReducer

from data_harvesting.environment import make_metrics_spec


def test_data_collection_metric_spec_matches_current_environment_contract() -> None:
    metrics_spec = make_metrics_spec()

    assert tuple(metric.key for metric in metrics_spec) == (
        "avg_reward",
        "max_reward",
        "sum_reward",
        "avg_collection_time",
        "episode_duration",
        "completion_time",
        "all_collected",
        "num_collected",
        "num_dead",
        "cause",
    )
    assert all(
        isinstance(metric, ScalarMetricSpec) and metric.reducer is ScalarReducer.MEAN
        for metric in metrics_spec[:-1]
    )

    cause_metric = metrics_spec[-1]
    assert isinstance(cause_metric, CategoricalMetricSpec)
    assert cause_metric.resolved_output_prefix == "end_cause"
    assert cause_metric.value_labels == {
        0: "NONE",
        1: "TIMEOUT",
        2: "ALL_COLLECTED",
        3: "STALLED",
        4: "ALL_AGENTS_INACTIVE",
    }
