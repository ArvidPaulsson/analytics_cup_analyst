"""
Example usage of the custom aggregation tooling, inspired by the
`Aggregating_Dynamic_Events_Tutorial.ipynb` notebook.

This script loads a sample dynamic events CSV, defines a couple of bespoke
contexts and metrics, runs the aggregation workflow, and writes the result to
``custom_aggregates.csv`` in the same directory.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from custom_aggregator.config import (
    ColumnFilter,
    ContextDefinition,
    MetricDefinition,
    OutputConfig,
)
from custom_aggregator.config import AggregatorConfig
from custom_aggregator.aggregator import CustomAggregationTool


def main() -> None:
    repo_root = PROJECT_ROOT.parent
    events_path = (
        repo_root
        / "opendata"
        / "data"
        / "matches"
        / "1886347"
        / "1886347_dynamic_events.csv"
    )

    contexts = {
        "custom": {
            "all_possessions": ContextDefinition(
                filters=[
                    ColumnFilter(
                        column="event_type", op="eq", value="player_possession"
                    ),
                    ColumnFilter(
                        column="team_in_possession_phase_type", op="eq", value="finish"
                    ),
                    ColumnFilter(column="separation_start", op="ge", value=5),
                ]
            )
        }
    }

    metrics = {
        "custom": {
            "count": MetricDefinition(builtin="count"),
            "avg_duration": MetricDefinition(builtin="mean", column="duration"),
            "avg_distance_covered": MetricDefinition(
                builtin="mean", column="distance_covered"
            ),
        }
    }

    config = AggregatorConfig(
        events_path=events_path,
        group_by=["player_id", "player_name"],
        contexts=contexts,
        metrics=metrics,
        aggregator_module_path=(
            repo_root
            / "opendata"
            / "resources"
            / "Tutorials"
            / "[Advanced] Aggregating Dynamic Events"
            / "DynamicEventsAggregator.py"
        ),
        output=OutputConfig(
            path=Path(__file__).with_name("custom_aggregates.csv"), format="csv"
        ),
    )

    tool = CustomAggregationTool(config=config)
    result = tool.run()

    print("Custom aggregates preview:")
    print(result.head())


if __name__ == "__main__":
    main()
