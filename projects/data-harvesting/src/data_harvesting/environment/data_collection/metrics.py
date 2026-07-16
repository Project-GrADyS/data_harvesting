from __future__ import annotations

from data_harvesting.environment.environment import EndCause
from rl_core import CategoricalMetricSpec, ScalarMetricSpec, ScalarReducer


def make_data_collection_metrics_spec() -> tuple[ScalarMetricSpec | CategoricalMetricSpec, ...]:
    return (
        ScalarMetricSpec(key="avg_reward", reducer=ScalarReducer.MEAN),
        ScalarMetricSpec(key="max_reward", reducer=ScalarReducer.MEAN),
        ScalarMetricSpec(key="sum_reward", reducer=ScalarReducer.MEAN),
        ScalarMetricSpec(key="avg_collection_time", reducer=ScalarReducer.MEAN),
        ScalarMetricSpec(key="episode_duration", reducer=ScalarReducer.MEAN),
        ScalarMetricSpec(key="completion_time", reducer=ScalarReducer.MEAN),
        ScalarMetricSpec(key="all_collected", reducer=ScalarReducer.MEAN),
        ScalarMetricSpec(key="num_collected", reducer=ScalarReducer.MEAN),
        ScalarMetricSpec(key="num_dead", reducer=ScalarReducer.MEAN),
        CategoricalMetricSpec(
            key="cause",
            output_prefix="end_cause",
            value_labels={cause.value: cause.name for cause in EndCause},
        ),
    )
