"""
Pressing event data loading and extraction utilities.

This module provides functions to extract pressing events from dynamic events data,
filter by type, and extract pressing chains.
"""

from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
from .data_loader import load_dynamic_events


# Pressing event types and subtypes
PRESSING_EVENT_TYPE = "on_ball_engagement"
PRESSING_SUBTYPES = ["pressing", "pressure", "recovery_press", "counter_press"]


def extract_pressing_events(
    dynamic_events: pd.DataFrame,
    event_types: Optional[List[str]] = None,
    event_subtypes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract pressing events from dynamic events data.

    Parameters
    ----------
    dynamic_events : pd.DataFrame
        Dynamic events DataFrame
    event_types : list of str, optional
        Event types to filter. If None, uses default pressing event type.
    event_subtypes : list of str, optional
        Event subtypes to filter. If None, uses all pressing subtypes.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only pressing events
    """
    if event_types is None:
        event_types = [PRESSING_EVENT_TYPE]

    if event_subtypes is None:
        event_subtypes = PRESSING_SUBTYPES

    # Filter for pressing events
    pressing_events = dynamic_events[
        (dynamic_events["event_type"] == PRESSING_EVENT_TYPE)
        & (dynamic_events["event_subtype"].isin(event_subtypes))
    ].copy()

    return pressing_events


def load_pressing_events(
    match_id: int,
    data_dir: Optional[Union[str, Path]] = None,
    use_github: bool = False,
    event_types: Optional[List[str]] = None,
    event_subtypes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load and filter pressing events for a match.

    Parameters
    ----------
    match_id : int
        Match ID
    data_dir : str or Path, optional
        Local directory containing match data
    use_github : bool, default False
        If True, load from GitHub. If False, load from local files.
    event_types : list of str, optional
        Event types to filter. If None, uses default pressing event type.
    event_subtypes : list of str, optional
        Event subtypes to filter. If None, uses all pressing subtypes.

    Returns
    -------
    pd.DataFrame
        Pressing events DataFrame
    """
    dynamic_events = load_dynamic_events(match_id, data_dir, use_github)
    return extract_pressing_events(dynamic_events, event_types, event_subtypes)


def extract_pressing_chains(pressing_events: pd.DataFrame) -> pd.DataFrame:
    """
    Extract pressing chain information from pressing events.

    Parameters
    ----------
    pressing_events : pd.DataFrame
        Pressing events DataFrame

    Returns
    -------
    pd.DataFrame
        Pressing events with chain information grouped
    """
    # Filter events that are part of a pressing chain
    chain_events = pressing_events[pressing_events["pressing_chain"].notna()].copy()

    # Sort by chain and index within chain
    if len(chain_events) > 0:
        chain_events = chain_events.sort_values(
            ["pressing_chain", "pressing_chain_index", "index_in_pressing_chain"]
        )

    return chain_events


def get_pressing_chain_summary(pressing_events: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for pressing chains.

    Parameters
    ----------
    pressing_events : pd.DataFrame
        Pressing events DataFrame

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with chain statistics
    """
    chain_events = extract_pressing_chains(pressing_events)

    if len(chain_events) == 0:
        return pd.DataFrame()

    # Group by pressing chain
    chain_summary = (
        chain_events.groupby("pressing_chain")
        .agg(
            {
                "frame_start": "min",
                "frame_end": "max",
                "pressing_chain_length": "first",
                "pressing_chain_end_type": "first",
                "player_id": "count",  # Number of events in chain
                "team_id": "first",
                "team_shortname": "first",
                "x_start": "mean",
                "y_start": "mean",
                "x_end": "mean",
                "y_end": "mean",
            }
        )
        .reset_index()
    )

    chain_summary = chain_summary.rename(
        columns={
            "player_id": "n_events",
            "frame_start": "chain_start_frame",
            "frame_end": "chain_end_frame",
        }
    )

    return chain_summary


def filter_pressing_by_phase_type(
    pressing_events: pd.DataFrame,
    phase_type: Optional[str] = None,
    defensive_phase_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter pressing events by phase type context.

    Parameters
    ----------
    pressing_events : pd.DataFrame
        Pressing events DataFrame
    phase_type : str, optional
        Filter by team in possession phase type (e.g., 'build_up', 'create', 'finish')
    defensive_phase_type : str, optional
        Filter by team out of possession phase type (e.g., 'high_block', 'medium_block', 'low_block')

    Returns
    -------
    pd.DataFrame
        Filtered pressing events
    """
    filtered = pressing_events.copy()

    if phase_type is not None:
        filtered = filtered[filtered["team_in_possession_phase_type"] == phase_type]

    if defensive_phase_type is not None:
        filtered = filtered[
            filtered["team_out_of_possession_phase_type"] == defensive_phase_type
        ]

    return filtered


def get_pressing_event_outcomes(pressing_events: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze outcomes of pressing events.

    Parameters
    ----------
    pressing_events : pd.DataFrame
        Pressing events DataFrame

    Returns
    -------
    pd.DataFrame
        Summary of pressing outcomes
    """
    outcomes = []

    # Check for various outcome indicators
    if "associated_player_possession_end_type" in pressing_events.columns:
        outcome_counts = pressing_events[
            "associated_player_possession_end_type"
        ].value_counts()
        outcomes.append(outcome_counts)

    # Check for possession loss indicators
    if "team_possession_loss_in_phase" in pressing_events.columns:
        possession_loss = pressing_events["team_possession_loss_in_phase"].sum()
        outcomes.append(pd.Series({"possession_loss": possession_loss}))

    # Check for shots/goals
    if "lead_to_shot" in pressing_events.columns:
        lead_to_shot = pressing_events["lead_to_shot"].sum()
        outcomes.append(pd.Series({"lead_to_shot": lead_to_shot}))

    if "lead_to_goal" in pressing_events.columns:
        lead_to_goal = pressing_events["lead_to_goal"].sum()
        outcomes.append(pd.Series({"lead_to_goal": lead_to_goal}))

    if outcomes:
        return pd.concat(outcomes, axis=0).to_frame("count")
    else:
        return pd.DataFrame()


def get_pressing_by_location(
    pressing_events: pd.DataFrame, zone_column: str = "third_start"
) -> pd.DataFrame:
    """
    Get pressing events grouped by location/zone.

    Parameters
    ----------
    pressing_events : pd.DataFrame
        Pressing events DataFrame
    zone_column : str, default 'third_start'
        Column to use for zone grouping (e.g., 'third_start', 'channel_start', 'penalty_area_start')

    Returns
    -------
    pd.DataFrame
        Summary of pressing by location
    """
    if zone_column not in pressing_events.columns:
        raise ValueError(f"Column '{zone_column}' not found in pressing events")

    location_summary = (
        pressing_events.groupby(zone_column)
        .agg(
            {
                "event_id": "count",
                "player_id": "nunique",
                "team_id": "first",
                "team_shortname": "first",
                "x_start": "mean",
                "y_start": "mean",
            }
        )
        .reset_index()
    )

    location_summary = location_summary.rename(
        columns={
            "event_id": "n_pressing_events",
            "player_id": "n_unique_players",
            zone_column: "zone",
        }
    )

    return location_summary
