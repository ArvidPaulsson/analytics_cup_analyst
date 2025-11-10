from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

from .config import MetricDefinition, MetricGroupConfig


SAFE_EVAL_BUILTINS = {
    "__import__": __import__,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "round": round,
}

SAFE_EVAL_GLOBALS = {"__builtins__": SAFE_EVAL_BUILTINS, "np": np, "pd": pd}

MetricCallable = Callable[[pd.DataFrame], object]


def _require_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column is None:
        raise ValueError("Metric requires a column but none was provided.")
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not present in DataFrame.")
    return df[column]


REGISTERED_METRICS: Dict[str, Callable[[pd.DataFrame, MetricDefinition], object]] = {
    "count": lambda df, _: len(df),
    "sum": lambda df, spec: _require_column(df, spec.column).sum(),
    "mean": lambda df, spec: _require_column(df, spec.column).mean(),
    "median": lambda df, spec: _require_column(df, spec.column).median(),
    "min": lambda df, spec: _require_column(df, spec.column).min(),
    "max": lambda df, spec: _require_column(df, spec.column).max(),
    "std": lambda df, spec: _require_column(df, spec.column).std(),
    "nunique": lambda df, spec: _require_column(df, spec.column).nunique(),
}


def build_metric_groups(
    metric_config: MetricGroupConfig,
) -> Dict[str, Dict[str, MetricCallable]]:
    """
    Translate metric configuration into callable groups.

    Returns:
        ``{group: {metric_name: callable}}`` structure compatible with
        ``DynamicEventAggregator``.
    """

    metric_groups: Dict[str, Dict[str, MetricCallable]] = {}

    for group_name, metrics in metric_config.items():
        metric_groups[group_name] = {}
        for metric_name, definition in metrics.items():
            if definition.builtin:
                builtin_id = definition.builtin.lower()
                if builtin_id not in REGISTERED_METRICS:
                    raise KeyError(
                        f"Unknown builtin metric '{definition.builtin}'. "
                        f"Valid options: {sorted(REGISTERED_METRICS)}"
                    )

                def builtin_metric(
                    df: pd.DataFrame,
                    *,
                    _impl=REGISTERED_METRICS[builtin_id],
                    _definition=definition,
                ) -> object:
                    return _impl(df, _definition)

                metric_groups[group_name][metric_name] = builtin_metric
                continue

            expression = definition.expression
            if not expression:
                raise ValueError(
                    f"Metric '{metric_name}' in group '{group_name}' lacks "
                    "a builtin and expression."
                )
            compiled = compile(expression, "<metric-expression>", "eval")

            def expression_metric(df: pd.DataFrame, *, _code=compiled) -> object:
                local_ns = {"df": df}
                return eval(_code, SAFE_EVAL_GLOBALS, local_ns)

            metric_groups[group_name][metric_name] = expression_metric

    return metric_groups
