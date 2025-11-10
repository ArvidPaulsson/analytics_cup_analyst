from __future__ import annotations

from typing import Dict

import pandas as pd

from .config import ColumnFilter, ContextDefinition, ContextGroupConfig


def _apply_filter(df: pd.DataFrame, flt: ColumnFilter) -> pd.Series:
    series = df[flt.column]
    op = flt.op.lower()
    value = flt.value

    if op == "eq":
        return series == value
    if op == "ne":
        return series != value
    if op == "gt":
        return series > value
    if op == "ge":
        return series >= value
    if op == "lt":
        return series < value
    if op == "le":
        return series <= value
    if op == "in":
        return series.isin(value if isinstance(value, (list, tuple, set)) else [value])
    if op == "not_in":
        return ~series.isin(value if isinstance(value, (list, tuple, set)) else [value])
    if op == "contains":
        return series.astype(str).str.contains(str(value), na=False)
    if op == "not_contains":
        return ~series.astype(str).str.contains(str(value), na=False)
    raise ValueError(f"Unsupported operator '{flt.op}' in filter.")


def build_context_groups(
    df: pd.DataFrame, context_config: ContextGroupConfig
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Construct the custom context groups expected by ``DynamicEventAggregator``.

    Args:
        df: Dynamic events DataFrame.
        context_config: Nested dictionary mapping context group -> context name ->
            ``ContextDefinition``.

    Returns:
        Dictionary matching the signature required by the aggregator:
        ``{group: {context_name: boolean_mask}}``.
    """

    context_groups: Dict[str, Dict[str, pd.Series]] = {}

    for group_name, contexts in context_config.items():
        context_groups[group_name] = {}
        for context_name, definition in contexts.items():
            mask = pd.Series(True, index=df.index)
            for flt in definition.filters:
                mask &= _apply_filter(df, flt)
            if definition.query:
                query_result = df.eval(definition.query, engine="python")
                if query_result.dtype != bool:
                    raise ValueError(
                        f"Query for context '{context_name}' did not return boolean."
                    )
                mask &= query_result
            context_groups[group_name][context_name] = mask

    return context_groups
