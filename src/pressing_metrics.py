"""
Pressing metrics calculation utilities.

This module provides functions to calculate comprehensive metrics
for pressing events grouped by defensive phase type and other dimensions.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np


def calculate_pressing_metrics_by_defensive_phase(
    pressing_events: pd.DataFrame,
    group_by_team: bool = False,
    phase_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate comprehensive pressing metrics grouped by team_out_of_possession_phase_type.

    This function aggregates pressing events by defensive phase type (e.g., high_block,
    medium_block, low_block) and calculates various metrics including:
    - Count metrics (total events, unique players)
    - Success metrics (possession loss rate, turnover rate)
    - Outcome metrics (shots, goals)
    - Spatial metrics (average location, pitch zones)
    - Intensity metrics (duration, speed, separation)
    - Chain metrics (chain length, participation)
    - Effectiveness metrics (danger reduction, backward forcing)

    Parameters
    ----------
    pressing_events : pd.DataFrame
        Pressing events DataFrame with phase information linked
    group_by_team : bool, default False
        If True, also group by team_id to get team-specific metrics
    phase_column : str, optional
        Name of the defensive phase type column. If None, will try to detect
        from available columns (team_out_of_possession_phase_type or
        phase_team_out_of_possession_phase_type).

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics grouped by defensive phase type
    """
    if len(pressing_events) == 0:
        return pd.DataFrame()

    # Determine the phase column name
    if phase_column is None:
        if "phase_team_out_of_possession_phase_type" in pressing_events.columns:
            phase_column = "phase_team_out_of_possession_phase_type"
        elif "team_out_of_possession_phase_type" in pressing_events.columns:
            phase_column = "team_out_of_possession_phase_type"
        else:
            raise ValueError(
                "No defensive phase type column found. Expected one of: "
                "'team_out_of_possession_phase_type' or "
                "'phase_team_out_of_possession_phase_type'"
            )

    # Determine grouping columns
    group_cols = [phase_column]
    if group_by_team and "team_id" in pressing_events.columns:
        group_cols.append("team_id")
        if "team_shortname" in pressing_events.columns:
            group_cols.append("team_shortname")

    # Filter out events without defensive phase type
    events_with_phase = pressing_events[pressing_events[phase_column].notna()].copy()

    if len(events_with_phase) == 0:
        return pd.DataFrame()

    # Initialize aggregation dictionary
    agg_dict: Dict[str, List[str]] = {
        "event_id": ["count"],
        "player_id": ["nunique"],
    }

    # Count metrics
    metrics = {}

    # Success/Outcome metrics
    if "team_possession_loss_in_phase" in events_with_phase.columns:
        agg_dict["team_possession_loss_in_phase"] = ["sum", "mean"]
        metrics["possession_loss_count"] = "team_possession_loss_in_phase_sum"
        metrics["possession_loss_rate"] = "team_possession_loss_in_phase_mean"

    if "lead_to_shot" in events_with_phase.columns:
        agg_dict["lead_to_shot"] = ["sum", "mean"]
        metrics["shots_resulting"] = "lead_to_shot_sum"
        metrics["shot_rate"] = "lead_to_shot_mean"

    if "lead_to_goal" in events_with_phase.columns:
        agg_dict["lead_to_goal"] = ["sum", "mean"]
        metrics["goals_resulting"] = "lead_to_goal_sum"
        metrics["goal_rate"] = "lead_to_goal_mean"

    # Duration metrics
    if "duration" in events_with_phase.columns:
        agg_dict["duration"] = ["mean", "median", "std", "min", "max"]
        metrics["avg_duration"] = "duration_mean"
        metrics["median_duration"] = "duration_median"

    # Spatial metrics
    if "x_start" in events_with_phase.columns:
        agg_dict["x_start"] = ["mean", "std"]
        agg_dict["y_start"] = ["mean", "std"]
        metrics["avg_x_location"] = "x_start_mean"
        metrics["avg_y_location"] = "y_start_mean"

    # Intensity metrics
    if "speed_avg" in events_with_phase.columns:
        agg_dict["speed_avg"] = ["mean", "median", "max"]
        metrics["avg_speed"] = "speed_avg_mean"
        metrics["max_speed"] = "speed_avg_max"

    if "separation_start" in events_with_phase.columns:
        agg_dict["separation_start"] = ["mean", "median", "min"]
        metrics["avg_separation"] = "separation_start_mean"
        metrics["min_separation"] = "separation_start_min"

    # Chain metrics
    if "pressing_chain" in events_with_phase.columns:
        if "pressing_chain_length" in events_with_phase.columns:
            agg_dict["pressing_chain_length"] = ["mean", "median", "max"]
            metrics["avg_chain_length"] = "pressing_chain_length_mean"
            metrics["max_chain_length"] = "pressing_chain_length_max"

    # Effectiveness metrics
    if "stop_possession_danger" in events_with_phase.columns:
        agg_dict["stop_possession_danger"] = ["sum", "mean"]
        metrics["danger_stopped_count"] = "stop_possession_danger_sum"
        metrics["danger_stopped_rate"] = "stop_possession_danger_mean"

    if "reduce_possession_danger" in events_with_phase.columns:
        agg_dict["reduce_possession_danger"] = ["sum", "mean"]
        metrics["danger_reduced_count"] = "reduce_possession_danger_sum"
        metrics["danger_reduced_rate"] = "reduce_possession_danger_mean"

    if "force_backward" in events_with_phase.columns:
        agg_dict["force_backward"] = ["sum", "mean"]
        metrics["backward_forced_count"] = "force_backward_sum"
        metrics["backward_forced_rate"] = "force_backward_mean"

    # Calculate aggregated metrics
    result = events_with_phase.groupby(group_cols).agg(agg_dict).reset_index()

    # Store the original group column names before any transformations
    original_group_cols = group_cols.copy()

    # Flatten column names (but preserve group columns)
    # Handle both tuple and string column names
    flattened_columns = []
    for col in result.columns.values:
        if isinstance(col, tuple):
            # Join tuple elements with underscore, filter out empty strings
            flattened = "_".join(str(c) for c in col if c)
            flattened_columns.append(flattened.strip("_") if flattened else str(col))
        else:
            flattened_columns.append(str(col))
    result.columns = flattened_columns

    # Rename columns for clarity (but don't rename group columns)
    # The metrics dict maps {desired_name: flattened_column_name}
    # But rename() needs {old_name: new_name}, so we need to reverse it
    rename_dict = {
        "event_id_count": "total_pressing_events",
        "player_id_nunique": "unique_pressing_players",
    }
    # Reverse the metrics dictionary: {old_name: new_name}
    for new_name, old_name in metrics.items():
        if old_name in result.columns and old_name not in group_cols:
            rename_dict[old_name] = new_name

    # Only rename columns that exist and are not group columns
    safe_rename_dict = {
        old_name: new_name
        for old_name, new_name in rename_dict.items()
        if old_name in result.columns and old_name not in group_cols
    }
    result = result.rename(columns=safe_rename_dict)

    # Update group_cols to match actual column names in result (in case they changed)
    group_cols = [col for col in original_group_cols if col in result.columns]

    # Calculate chain participation rate after grouping
    if "pressing_chain" in events_with_phase.columns:
        # Use original_group_cols for grouping to ensure we group by the same columns
        chain_participation = (
            events_with_phase.groupby(original_group_cols)["pressing_chain"]
            .apply(lambda x: (x.notna().sum() / len(x)) * 100)
            .reset_index(name="chain_participation_rate")
        )
        # Verify all merge keys exist in both DataFrames
        # Use the current group_cols (which match result) for merging
        missing_in_result = [col for col in group_cols if col not in result.columns]
        missing_in_chain = [
            col for col in original_group_cols if col not in chain_participation.columns
        ]

        if missing_in_result or missing_in_chain:
            # If columns are missing, skip the merge and add a default value
            result["chain_participation_rate"] = 0.0
        else:
            # Check if we can merge directly
            if group_cols == original_group_cols:
                # Column names match, proceed with merge
                result = result.merge(chain_participation, on=group_cols, how="left")
            else:
                # Column names might have changed, create a mapping
                # Since we preserved group columns, they should match
                # But if they don't, merge on index alignment
                try:
                    result = result.merge(
                        chain_participation,
                        left_on=group_cols,
                        right_on=original_group_cols,
                        how="left",
                    )
                except (KeyError, ValueError):
                    # Fallback: add default value if merge fails
                    result["chain_participation_rate"] = 0.0
            # Fill NaN values with 0 for groups that had no chain events
            if "chain_participation_rate" in result.columns:
                result["chain_participation_rate"] = result[
                    "chain_participation_rate"
                ].fillna(0.0)

    # Calculate additional derived metrics
    if "total_pressing_events" in result.columns:
        # Events per player
        if "unique_pressing_players" in result.columns:
            result["events_per_player"] = (
                result["total_pressing_events"] / result["unique_pressing_players"]
            )

    # Calculate success rates as percentages (skip chain_participation_rate as it's already a percentage)
    # Ensure we only process string column names
    rate_columns = [
        col
        for col in result.columns
        if isinstance(col, str)
        and col.endswith("_rate")
        and col != "chain_participation_rate"
    ]
    for col in rate_columns:
        if col in result.columns and result[col].dtype in [float, int]:
            result[col] = result[col] * 100

    return result


def calculate_pressing_metrics_by_subtype_and_phase(
    pressing_events: pd.DataFrame,
    phase_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate pressing metrics grouped by event subtype and defensive phase type.

    This provides a more granular view of how different types of pressing
    (pressure, pressing, recovery_press, counter_press) perform across
    different defensive phase types.

    Parameters
    ----------
    pressing_events : pd.DataFrame
        Pressing events DataFrame with phase information linked
    phase_column : str, optional
        Name of the defensive phase type column. If None, will try to detect
        from available columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics grouped by event_subtype and defensive phase type
    """
    if len(pressing_events) == 0:
        return pd.DataFrame()

    # Determine the phase column name
    if phase_column is None:
        if "phase_team_out_of_possession_phase_type" in pressing_events.columns:
            phase_column = "phase_team_out_of_possession_phase_type"
        elif "team_out_of_possession_phase_type" in pressing_events.columns:
            phase_column = "team_out_of_possession_phase_type"
        else:
            raise ValueError(
                "No defensive phase type column found. Expected one of: "
                "'team_out_of_possession_phase_type' or "
                "'phase_team_out_of_possession_phase_type'"
            )

    # Add event_subtype to grouping
    events_with_both = pressing_events[
        (pressing_events[phase_column].notna())
        & (pressing_events["event_subtype"].notna())
    ].copy()

    if len(events_with_both) == 0:
        return pd.DataFrame()

    # Use the main function with event_subtype in grouping
    group_cols = ["event_subtype", phase_column]
    if "team_id" in events_with_both.columns:
        group_cols.append("team_id")
        if "team_shortname" in events_with_both.columns:
            group_cols.append("team_shortname")

    # Use similar aggregation as main function but with event_subtype
    agg_dict: Dict[str, List[str]] = {
        "event_id": ["count"],
        "player_id": ["nunique"],
    }

    if "team_possession_loss_in_phase" in events_with_both.columns:
        agg_dict["team_possession_loss_in_phase"] = ["sum", "mean"]

    if "lead_to_shot" in events_with_both.columns:
        agg_dict["lead_to_shot"] = ["sum", "mean"]

    if "lead_to_goal" in events_with_both.columns:
        agg_dict["lead_to_goal"] = ["sum", "mean"]

    if "duration" in events_with_both.columns:
        agg_dict["duration"] = ["mean", "median"]

    if "speed_avg" in events_with_both.columns:
        agg_dict["speed_avg"] = ["mean", "max"]

    result = events_with_both.groupby(group_cols).agg(agg_dict).reset_index()

    # Flatten column names (handle both tuple and string column names)
    flattened_columns = []
    for col in result.columns.values:
        if isinstance(col, tuple):
            # Join tuple elements with underscore, filter out empty strings
            flattened = "_".join(str(c) for c in col if c)
            flattened_columns.append(flattened.strip("_") if flattened else str(col))
        else:
            flattened_columns.append(str(col))
    result.columns = flattened_columns

    # Rename for clarity
    result = result.rename(
        columns={
            "event_id_count": "total_events",
            "player_id_nunique": "unique_players",
            "team_possession_loss_in_phase_sum": "possession_loss_count",
            "team_possession_loss_in_phase_mean": "possession_loss_rate",
            "lead_to_shot_sum": "shots_resulting",
            "lead_to_shot_mean": "shot_rate",
            "lead_to_goal_sum": "goals_resulting",
            "lead_to_goal_mean": "goal_rate",
            "duration_mean": "avg_duration",
            "speed_avg_mean": "avg_speed",
        }
    )

    # Convert rates to percentages (ensure we only process string column names)
    rate_cols = [
        col for col in result.columns if isinstance(col, str) and col.endswith("_rate")
    ]
    for col in rate_cols:
        if col in result.columns and result[col].dtype in [float, int]:
            result[col] = result[col] * 100

    return result


def get_pressing_effectiveness_score(
    pressing_metrics: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Calculate a composite pressing effectiveness score.

    This combines multiple metrics into a single effectiveness score
    that can be used to compare pressing performance across different
    defensive phase types.

    Parameters
    ----------
    pressing_metrics : pd.DataFrame
        DataFrame with pressing metrics (output from calculate_pressing_metrics_by_defensive_phase)
    weights : dict, optional
        Dictionary of metric names and their weights for the composite score.
        If None, uses default weights.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'pressing_effectiveness_score' column
    """
    if weights is None:
        weights = {
            "possession_loss_rate": 0.3,
            "danger_stopped_rate": 0.25,
            "danger_reduced_rate": 0.2,
            "backward_forced_rate": 0.15,
            "shot_rate": 0.1,
        }

    result = pressing_metrics.copy()

    # Normalize metrics to 0-1 scale for scoring
    score = 0.0
    total_weight = 0.0

    for metric, weight in weights.items():
        if metric in result.columns:
            # Normalize to 0-1 scale (assuming max value is 100 for rates)
            normalized = result[metric] / 100.0
            score += normalized * weight
            total_weight += weight

    # Normalize by total weight
    if total_weight > 0:
        result["pressing_effectiveness_score"] = score / total_weight * 100
    else:
        result["pressing_effectiveness_score"] = 0.0

    return result
