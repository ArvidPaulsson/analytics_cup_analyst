from __future__ import annotations

import importlib
from importlib.machinery import SourceFileLoader
from typing import Dict, Type

import pandas as pd

from .config import AggregatorConfig
from .contexts import build_context_groups
from .io import aggregate_tracking, read_dataframe, select_columns, write_dataframe
from .metrics import build_metric_groups


def _load_dynamic_event_aggregator(
    config: AggregatorConfig,
) -> Type:
    """
    Retrieve the ``DynamicEventAggregator`` class either from a module on sys.path
    or a user-specified file path.
    """

    if config.aggregator_module_path:
        loader = SourceFileLoader(
            "custom_dynamic_event_aggregator", str(config.aggregator_module_path)
        )
        module = loader.load_module()  # type: ignore[deprecated-method]
        return getattr(module, "DynamicEventAggregator")

    module = importlib.import_module("DynamicEventsAggregator")
    return getattr(module, "DynamicEventAggregator")


class CustomAggregationTool:
    """High-level orchestrator for building event aggregates."""

    def __init__(self, config: AggregatorConfig):
        self.config = config

    def _prepare_contexts(self, events_df: pd.DataFrame) -> Dict:
        if not self.config.contexts:
            return {}
        return build_context_groups(events_df, self.config.contexts)

    def _prepare_metrics(self) -> Dict:
        if not self.config.metrics:
            return {}
        return build_metric_groups(self.config.metrics)

    def run(self) -> pd.DataFrame:
        """Execute the aggregation workflow and return the resulting DataFrame."""

        events_df = read_dataframe(self.config.events_path)
        context_groups = self._prepare_contexts(events_df)
        metric_groups = self._prepare_metrics()

        aggregator_cls = _load_dynamic_event_aggregator(self.config)
        aggregator_kwargs = {"df": events_df}
        if context_groups:
            aggregator_kwargs["custom_context_groups"] = context_groups
        if metric_groups:
            aggregator_kwargs["custom_metric_groups"] = metric_groups

        aggregator = aggregator_cls(**aggregator_kwargs)
        aggregates = aggregator.generate_aggregates(
            group_by=self.config.group_by, aggregate_type=self.config.aggregate_type
        )

        if self.config.tracking:
            tracking_cfg = self.config.tracking
            tracking_df = read_dataframe(tracking_cfg.path)
            tracking_df = select_columns(tracking_df, tracking_cfg.columns)
            tracking_df = aggregate_tracking(
                tracking_df, tracking_cfg.join_on, tracking_cfg.aggregations
            )
            aggregates = aggregates.merge(
                tracking_df,
                on=tracking_cfg.join_on,
                how=tracking_cfg.how,
            )

        if self.config.output:
            write_dataframe(
                aggregates,
                self.config.output.path,
                self.config.output.format,
            )

        return aggregates


