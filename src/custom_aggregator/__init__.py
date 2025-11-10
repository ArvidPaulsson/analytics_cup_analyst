"""
Declarative tooling for building custom dynamic-event aggregates.

Exposes the primary orchestration class (`CustomAggregationTool`) together with
configuration models so analysts can script or configure bespoke aggregation
workflows.
"""

from .aggregator import CustomAggregationTool
from .config import (
    AggregatorConfig,
    ContextDefinition,
    ContextGroupConfig,
    MetricDefinition,
    MetricGroupConfig,
    TrackingMergeConfig,
)

__all__ = [
    "AggregatorConfig",
    "ContextDefinition",
    "ContextGroupConfig",
    "CustomAggregationTool",
    "MetricDefinition",
    "MetricGroupConfig",
    "TrackingMergeConfig",
]
