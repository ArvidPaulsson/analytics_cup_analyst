"""
Configuration models for building custom aggregates with ``DynamicEventAggregator``.

Supported filter operators:
    - ``eq`` / ``ne``: equality / inequality
    - ``gt`` / ``ge`` / ``lt`` / ``le``: numeric comparisons
    - ``in`` / ``not_in``: membership against a collection
    - ``contains`` / ``not_contains``: substring search for object columns

Supported built-in metric identifiers (see ``metrics.REGISTERED_METRICS``):
    - ``count``
    - ``sum``
    - ``mean``
    - ``median``
    - ``min``
    - ``max``
    - ``std``
    - ``nunique``

Analysts can alternatively provide a Python expression via ``expression`` that
is evaluated with a ``df`` variable bound to the context DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ColumnFilter(BaseModel):
    column: str = Field(..., description="Name of the DataFrame column to filter")
    op: str = Field(
        ...,
        description=(
            "Filter operator. One of eq, ne, gt, ge, lt, le, in, not_in, "
            "contains, not_contains."
        ),
    )
    value: object = Field(..., description="Comparison value for the operator")


class ContextDefinition(BaseModel):
    """Defines a context mask constructed from column filters and/or a query."""

    filters: List[ColumnFilter] = Field(
        default_factory=list, description="List of column filters combined by AND"
    )
    query: Optional[str] = Field(
        default=None,
        description=(
            "Optional pandas-compatible boolean expression evaluated with df.eval. "
            "Should return a boolean Series that is AND-ed with the filters."
        ),
    )


ContextGroupConfig = Dict[str, Dict[str, ContextDefinition]]


class MetricDefinition(BaseModel):
    """Describes how to compute a metric for a context."""

    builtin: Optional[str] = Field(
        default=None, description="Optional identifier for a registered metric"
    )
    column: Optional[str] = Field(
        default=None,
        description="Optional column used by the builtin metric (e.g. sum, mean).",
    )
    expression: Optional[str] = Field(
        default=None,
        description=(
            "Python expression evaluated with `df` bound to the context DataFrame. "
            "Must return a scalar."
        ),
    )

    @model_validator(mode="after")
    def _validate_exclusive_fields(cls, values: MetricDefinition) -> MetricDefinition:
        builtin = values.builtin
        expression = values.expression
        if builtin and expression:
            raise ValueError("Provide either `builtin` or `expression`, not both.")
        if not builtin and not expression:
            raise ValueError("One of `builtin` or `expression` must be provided.")
        return values


MetricGroupConfig = Dict[str, Dict[str, MetricDefinition]]


class TrackingMergeConfig(BaseModel):
    """Configuration for merging tracking aggregates with event aggregates."""

    path: Path = Field(..., description="File containing tracking data (CSV/Parquet)")
    join_on: List[str] = Field(
        ..., description="Columns used to join tracking data onto aggregates"
    )
    columns: Optional[List[str]] = Field(
        default=None, description="Optional subset of columns to retain before join"
    )
    aggregations: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Optional column -> aggregation function mapping (e.g. {'speed': 'mean'})."
            " When provided, tracking data is grouped by join_on before merging."
        ),
    )
    how: str = Field(
        default="left",
        description="Join mode (passed to DataFrame.merge). Defaults to 'left'.",
    )


class OutputConfig(BaseModel):
    path: Path = Field(..., description="Destination file for aggregate output")
    format: str = Field(
        default="csv",
        description="Output format: one of 'csv', 'parquet', or 'json'.",
    )


class AggregatorConfig(BaseModel):
    """Top-level configuration for the custom aggregation tool."""

    events_path: Path = Field(..., description="Path to the dynamic events dataset")
    group_by: List[str] = Field(
        ...,
        description="Columns passed to DynamicEventAggregator.generate_aggregates.",
    )
    aggregate_type: str = Field(
        default="custom",
        description="Aggregate type to request from the DynamicEventAggregator.",
    )
    contexts: ContextGroupConfig = Field(
        default_factory=dict,
        description="Context group definitions mapped to context masks.",
    )
    metrics: MetricGroupConfig = Field(
        default_factory=dict,
        description="Metric group definitions mapped to callables.",
    )
    output: Optional[OutputConfig] = Field(
        default=None,
        description=("Optional configuration for writing the resulting aggregates."),
    )
    tracking: Optional[TrackingMergeConfig] = Field(
        default=None,
        description="Optional tracking dataset merge configuration.",
    )
    tracking_time_column: Optional[str] = Field(
        default=None,
        description=(
            "Optional timestamp column used when aligning tracking data externally."
            " Provided for completeness; not used directly by the base tool."
        ),
    )
    aggregator_module_path: Optional[Path] = Field(
        default=None,
        description=(
            "Optional filesystem path to a DynamicEventAggregator implementation."
            " If omitted, the import 'DynamicEventsAggregator.DynamicEventAggregator'"
            " is used (assuming it is on sys.path)."
        ),
    )
