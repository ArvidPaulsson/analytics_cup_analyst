"""
Streamlit application for configuring and executing custom event aggregates.

This UI wraps the ``CustomAggregationTool`` so analysts can interactively pick
contexts and metrics, run the aggregation, and inspect or download the results.
"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# Ensure the package source directory is importable when `streamlit run` is
# executed from the repository root.
APP_PATH = Path(__file__).resolve()
SRC_ROOT = APP_PATH.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REPO_ROOT = APP_PATH.parents[4]
DEFAULT_AGGREGATOR_MODULE = (
    REPO_ROOT
    / "opendata"
    / "resources"
    / "Tutorials"
    / "[Advanced] Aggregating Dynamic Events"
    / "DynamicEventsAggregator.py"
)

from custom_aggregator.aggregator import CustomAggregationTool
from custom_aggregator.config import (
    AggregatorConfig,
    ColumnFilter,
    ContextDefinition,
    MetricDefinition,
)

FILTER_OPERATORS = [
    "eq",
    "ne",
    "gt",
    "ge",
    "lt",
    "le",
    "in",
    "not_in",
    "contains",
    "not_contains",
]

BuiltinOptions = [
    "count",
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "nunique",
]


@dataclass
class FilterRow:
    column: str = ""
    op: str = "eq"
    value: str = ""


@dataclass
class ContextState:
    name: str
    filters: List[FilterRow] = field(default_factory=lambda: [FilterRow()])
    query: str = ""


@dataclass
class ContextGroupState:
    name: str
    contexts: List[ContextState] = field(default_factory=list)


@dataclass
class MetricRow:
    metric_name: str = ""
    builtin: str = ""
    column: str = ""
    expression: str = ""


@dataclass
class MetricGroupState:
    name: str
    metrics: List[MetricRow] = field(default_factory=lambda: [MetricRow()])


def _init_session_state() -> None:
    if "context_groups" not in st.session_state:
        st.session_state.context_groups: List[ContextGroupState] = [
            ContextGroupState(
                name="custom",
                contexts=[ContextState(name="example_context")],
            )
        ]
    if "metric_groups" not in st.session_state:
        st.session_state.metric_groups: List[MetricGroupState] = [
            MetricGroupState(name="custom")
        ]


def _literal_or_string(value: object) -> object:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float, bool, list, tuple, dict, set)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return value
    return value


def _editor_value_to_str(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
    except TypeError:
        pass
    return str(value)


@st.cache_data(show_spinner=False)
def _load_events(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Unsupported events file format. Use CSV or Parquet.")


def _discover_dynamic_event_files() -> List[str]:
    matches_root = REPO_ROOT / "opendata" / "data" / "matches"
    if not matches_root.exists():
        return []
    candidates = sorted(
        matches_root.rglob("*_dynamic_events.csv"), key=lambda p: p.name
    )
    return [str(path) for path in candidates]


def _render_context_builder(
    available_columns: List[str],
) -> Dict[str, Dict[str, ContextDefinition]]:
    st.subheader("Context Groups")
    context_groups = st.session_state.context_groups

    if st.button("➕ Add context group", type="secondary"):
        context_groups.append(
            ContextGroupState(
                name=f"group_{len(context_groups) + 1}",
                contexts=[ContextState(name="new_context")],
            )
        )

    built_groups: Dict[str, Dict[str, ContextDefinition]] = {}

    for idx, group in enumerate(context_groups):
        with st.expander(f"Group: {group.name}", expanded=True):
            new_group_name = st.text_input(
                "Group name",
                value=group.name,
                key=f"group-name-{idx}",
                help="Identifier passed to the aggregator",
            )
            group.name = new_group_name or group.name

            if st.button("➕ Add context", key=f"add-context-{idx}", type="secondary"):
                group.contexts.append(
                    ContextState(name=f"context_{len(group.contexts)+1}")
                )

            contexts_dict: Dict[str, ContextDefinition] = {}

            for c_idx, context in enumerate(group.contexts):
                with st.container():
                    cols = st.columns([3, 1])
                    context.name = cols[0].text_input(
                        "Context name",
                        value=context.name,
                        key=f"context-name-{idx}-{c_idx}",
                    )
                    remove_context = cols[1].button(
                        "🗑️ Remove",
                        key=f"remove-context-{idx}-{c_idx}",
                        type="secondary",
                    )

                    st.caption(
                        "Define filters combined with logical AND. Leave value blank to ignore."
                    )
                    filters_df = st.data_editor(
                        pd.DataFrame(
                            [
                                {"column": f.column, "op": f.op, "value": f.value}
                                for f in context.filters
                            ]
                        ),
                        num_rows="dynamic",
                        use_container_width=True,
                        key=f"filters-editor-{idx}-{c_idx}",
                        column_config={
                            "column": st.column_config.SelectboxColumn(
                                "Column",
                                options=[""] + available_columns,
                                required=False,
                            ),
                            "op": st.column_config.SelectboxColumn(
                                "Operator",
                                options=FILTER_OPERATORS,
                                required=True,
                            ),
                            "value": st.column_config.TextColumn("Value"),
                        },
                    )

                    context.filters = [
                        FilterRow(
                            column=row.get("column", ""),
                            op=row.get("op", "eq"),
                            value=_editor_value_to_str(row.get("value", "")),
                        )
                        for _, row in filters_df.iterrows()
                        if row.get("column") and row.get("op")
                    ]

                    context.query = st.text_area(
                        "Optional pandas query",
                        value=context.query,
                        key=f"context-query-{idx}-{c_idx}",
                        placeholder="e.g. (df.speed > 7) & (df.result == 'goal')",
                    )

                    if remove_context:
                        group.contexts.pop(c_idx)
                        st.experimental_rerun()

                if context.name:
                    filters = [
                        ColumnFilter(
                            column=f.column,
                            op=f.op,
                            value=_literal_or_string(f.value),
                        )
                        for f in context.filters
                        if f.column and f.op
                    ]
                    contexts_dict[context.name] = ContextDefinition(
                        filters=filters,
                        query=context.query or None,
                    )

            if st.button("🗑️ Delete group", key=f"remove-group-{idx}", type="secondary"):
                context_groups.pop(idx)
                st.experimental_rerun()

            if contexts_dict:
                built_groups[group.name] = contexts_dict

    return built_groups


def _render_metric_builder(
    available_columns: List[str],
) -> Dict[str, Dict[str, MetricDefinition]]:
    st.subheader("Metric Groups")
    metric_groups = st.session_state.metric_groups

    if st.button("➕ Add metric group", type="secondary"):
        metric_groups.append(
            MetricGroupState(
                name=f"metrics_{len(metric_groups) + 1}",
                metrics=[MetricRow()],
            )
        )

    built_groups: Dict[str, Dict[str, MetricDefinition]] = {}

    for idx, group in enumerate(metric_groups):
        with st.expander(f"Group: {group.name}", expanded=True):
            new_group_name = st.text_input(
                "Metric group name",
                value=group.name,
                key=f"metric-group-name-{idx}",
            )
            group.name = new_group_name or group.name

            st.caption(
                "Provide either builtin plus optional column, or a Python expression that references df."
            )
            metrics_df = st.data_editor(
                pd.DataFrame(
                    [
                        {
                            "metric_name": row.metric_name,
                            "builtin": row.builtin,
                            "column": row.column,
                            "expression": row.expression,
                        }
                        for row in group.metrics
                    ]
                ),
                num_rows="dynamic",
                use_container_width=True,
                key=f"metrics-editor-{idx}",
                column_config={
                    "metric_name": st.column_config.TextColumn("Metric name"),
                    "builtin": st.column_config.SelectboxColumn(
                        "Builtin id",
                        options=[""] + BuiltinOptions,
                        required=False,
                    ),
                    "column": st.column_config.SelectboxColumn(
                        "Column (for builtin)",
                        options=[""] + available_columns,
                        required=False,
                    ),
                    "expression": st.column_config.TextColumn(
                        "Python expression",
                        help="Alternative to builtin (mutually exclusive). Uses df variable.",
                    ),
                },
            )

            group.metrics = [
                MetricRow(
                    metric_name=_editor_value_to_str(row.get("metric_name", "")),
                    builtin=_editor_value_to_str(row.get("builtin", "")),
                    column=_editor_value_to_str(row.get("column", "")),
                    expression=_editor_value_to_str(row.get("expression", "")),
                )
                for _, row in metrics_df.iterrows()
                if row.get("metric_name")
            ]

            if st.button(
                "🗑️ Delete metric group",
                key=f"remove-metric-group-{idx}",
                type="secondary",
            ):
                metric_groups.pop(idx)
                st.experimental_rerun()

            if group.metrics:
                built_groups[group.name] = {}
                for row in group.metrics:
                    builtin = row.builtin.strip() or None
                    expression = row.expression.strip() or None
                    column = row.column.strip() or None
                    if builtin and expression:
                        st.warning(
                            f"Metric '{row.metric_name}' has both builtin and expression defined. "
                            "Only one will be used.",
                            icon="⚠️",
                        )
                    try:
                        built_groups[group.name][row.metric_name] = MetricDefinition(
                            builtin=builtin,
                            column=column,
                            expression=expression,
                        )
                    except Exception as exc:
                        st.error(
                            f"Failed to configure metric '{row.metric_name}': {exc}"
                        )

    return built_groups


def _build_config(
    events_path: str,
    group_by: List[str],
    contexts: Dict[str, Dict[str, ContextDefinition]],
    metrics: Dict[str, Dict[str, MetricDefinition]],
    aggregator_module_path: Optional[str],
) -> AggregatorConfig:
    cfg_kwargs = dict(
        events_path=Path(events_path),
        group_by=group_by,
        contexts=contexts,
        metrics=metrics,
    )
    if aggregator_module_path:
        cfg_kwargs["aggregator_module_path"] = Path(aggregator_module_path)
    return AggregatorConfig(**cfg_kwargs)


def main() -> None:
    st.set_page_config(
        page_title="Custom Aggregate Builder",
        layout="wide",
    )
    st.title("Custom Aggregate Builder")
    st.caption(
        "Configure context filters and metrics, then run the custom aggregation tool."
    )

    _init_session_state()

    with st.sidebar:
        st.header("Data & Aggregator")
        events_options = _discover_dynamic_event_files()
        events_path = st.selectbox(
            "Dynamic events file",
            options=events_options,
            index=0 if events_options else None,
            help="Select a CSV with dynamic events data.",
        )
        custom_events_path = st.text_input(
            "Or provide events path",
            value=events_path or "",
            help="Override with an absolute path to CSV/Parquet.",
        )
        events_path = custom_events_path.strip() or events_path

        aggregator_path_default = (
            str(DEFAULT_AGGREGATOR_MODULE) if DEFAULT_AGGREGATOR_MODULE.exists() else ""
        )
        aggregator_module_path = st.text_input(
            "DynamicEventAggregator module path",
            value=aggregator_path_default,
            help="Path to DynamicEventsAggregator.py implementation.",
        )

    events_df: Optional[pd.DataFrame] = None
    available_columns: List[str] = []
    if events_path:
        try:
            events_df = _load_events(events_path)
            available_columns = list(events_df.columns)
            st.success(f"Loaded events data with {len(events_df)} rows.", icon="✅")
        except Exception as exc:
            st.error(f"Failed to load events data: {exc}", icon="🚫")

    group_by = st.multiselect(
        "Group by columns",
        options=available_columns,
        default=[
            col for col in ("player_id", "player_name") if col in available_columns
        ],
        help="Columns passed to DynamicEventAggregator.generate_aggregates.",
    )

    contexts = _render_context_builder(available_columns)
    metrics = _render_metric_builder(available_columns)

    with st.expander("Configuration JSON preview", expanded=False):
        try:
            cfg_preview = _build_config(
                events_path=events_path or "",
                group_by=group_by,
                contexts=contexts,
                metrics=metrics,
                aggregator_module_path=aggregator_module_path or None,
            )
            st.code(json.dumps(cfg_preview.model_dump(mode="json"), indent=2))
        except Exception as exc:
            st.warning(f"Current configuration is invalid: {exc}")

    run_disabled = not (events_path and group_by and metrics)
    if run_disabled:
        st.info(
            "Select events data, choose group-by columns, and define at least one metric to run.",
            icon="ℹ️",
        )

    result_container = st.container()
    if st.button("🚀 Run aggregation", type="primary", disabled=run_disabled):
        try:
            config = _build_config(
                events_path=events_path,
                group_by=group_by,
                contexts=contexts,
                metrics=metrics,
                aggregator_module_path=aggregator_module_path or None,
            )
            tool = CustomAggregationTool(config=config)
            result_df = tool.run()
        except Exception as exc:
            result_container.error(f"Aggregation failed: {exc}")
        else:
            result_container.success(
                f"Aggregation complete. Result contains {len(result_df)} rows and {len(result_df.columns)} columns.",
                icon="✅",
            )
            result_container.dataframe(result_df, use_container_width=True)
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            result_container.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="custom_aggregates.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
