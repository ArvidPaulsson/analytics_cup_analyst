"""
Data merging utilities for combining pressing events with tracking data and phases of play.

This module provides functions to merge pressing events with tracking data
and phases of play into unified datasets for analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from .data_loader import load_match_data
from .pressing_data_loader import load_pressing_events, extract_pressing_events
from .tracking_loader import load_and_preprocess_tracking, get_tracking_for_event
from .phase_loader import load_phases_for_pressing_analysis, link_pressing_to_phase


def merge_pressing_with_tracking(
    pressing_events: pd.DataFrame,
    tracking_data: pd.DataFrame,
    window_before: int = 30,
    window_after: int = 30,
) -> pd.DataFrame:
    """
    Merge pressing events with tracking data.

    Parameters
    ----------
    pressing_events : pd.DataFrame
        Pressing events DataFrame (must have 'frame_start' and 'frame_end' columns)
    tracking_data : pd.DataFrame
        Preprocessed tracking data DataFrame
    window_before : int, default 30
        Number of frames before event start to include in tracking window
    window_after : int, default 30
        Number of frames after event end to include in tracking window

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with pressing events and associated tracking data
    """
    merged_data = []

    for _, event in pressing_events.iterrows():
        event_frame_start = event.get("frame_start")
        event_frame_end = event.get("frame_end", event_frame_start)

        # Get tracking data for this event
        event_tracking = get_tracking_for_event(
            tracking_data,
            event_frame_start,
            event_frame_end,
            window_before=window_before,
            window_after=window_after,
        )

        if len(event_tracking) > 0:
            # Add event information to tracking data
            event_tracking = event_tracking.copy()
            for col in pressing_events.columns:
                event_tracking[col] = event[col]

            merged_data.append(event_tracking)

    if len(merged_data) > 0:
        return pd.concat(merged_data, ignore_index=True)
    else:
        return pd.DataFrame()


def merge_pressing_with_phases(
    pressing_events: pd.DataFrame, phases_of_play: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge pressing events with phases of play data.

    Parameters
    ----------
    pressing_events : pd.DataFrame
        Pressing events DataFrame
    phases_of_play : pd.DataFrame
        Phases of play DataFrame

    Returns
    -------
    pd.DataFrame
        Pressing events with phase context added
    """
    return link_pressing_to_phase(pressing_events, phases_of_play)


def create_pressing_situation_dataset(
    pressing_events: pd.DataFrame,
    tracking_data: pd.DataFrame,
    phases_of_play: pd.DataFrame,
    window_before: int = 30,
    window_after: int = 30,
) -> pd.DataFrame:
    """
    Create a unified dataset combining pressing events, tracking data, and phases of play.

    Parameters
    ----------
    pressing_events : pd.DataFrame
        Pressing events DataFrame
    tracking_data : pd.DataFrame
        Preprocessed tracking data DataFrame
    phases_of_play : pd.DataFrame
        Phases of play DataFrame
    window_before : int, default 30
        Number of frames before event start to include in tracking window
    window_after : int, default 30
        Number of frames after event end to include in tracking window

    Returns
    -------
    pd.DataFrame
        Unified dataset with pressing events, tracking, and phase context
    """
    # First link pressing events to phases
    pressing_with_phases = merge_pressing_with_phases(pressing_events, phases_of_play)

    # Then merge with tracking data
    pressing_with_tracking = merge_pressing_with_tracking(
        pressing_with_phases,
        tracking_data,
        window_before=window_before,
        window_after=window_after,
    )

    return pressing_with_tracking


def create_pressing_dataset(
    match_id: int,
    data_dir: Optional[Union[str, Path]] = None,
    use_github: bool = False,
    load_tracking: bool = True,
    window_before: int = 30,
    window_after: int = 30,
    event_types: Optional[List[str]] = None,
    event_subtypes: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Create a unified pressing dataset for a match.

    This is the main function to load and merge all data sources for pressing analysis.

    Parameters
    ----------
    match_id : int
        Match ID
    data_dir : str or Path, optional
        Local directory containing match data
    use_github : bool, default False
        If True, load from GitHub. If False, load from local files.
    load_tracking : bool, default True
        If True, load and merge tracking data. If False, skip tracking data.
    window_before : int, default 30
        Number of frames before event start to include in tracking window
    window_after : int, default 30
        Number of frames after event end to include in tracking window
    event_types : list of str, optional
        Event types to filter. If None, uses default pressing event type.
    event_subtypes : list of str, optional
        Event subtypes to filter. If None, uses all pressing subtypes.

    Returns
    -------
    dict
        Dictionary containing:
        - 'pressing_events': Pressing events DataFrame
        - 'pressing_with_phases': Pressing events with phase context
        - 'pressing_with_tracking': Pressing events with tracking data (if load_tracking=True)
        - 'unified_dataset': Complete unified dataset (if load_tracking=True)
        - 'metadata': Match metadata
    """
    # Load all data
    match_data = load_match_data(
        match_id, data_dir, use_github, load_tracking=load_tracking
    )

    # Extract pressing events
    pressing_events = extract_pressing_events(
        match_data["dynamic_events"],
        event_types=event_types,
        event_subtypes=event_subtypes,
    )

    # Link to phases
    pressing_with_phases = merge_pressing_with_phases(
        pressing_events, match_data["phases_of_play"]
    )

    result = {
        "pressing_events": pressing_events,
        "pressing_with_phases": pressing_with_phases,
        "metadata": match_data["metadata"],
    }

    # Merge with tracking if requested
    if load_tracking and "tracking_data" in match_data:
        raw_tracking_data = match_data["tracking_data"]

        # Preprocess tracking data - check if it's already preprocessed
        # Raw tracking data from JSONL has nested structure that needs preprocessing
        if (
            "player_data" in raw_tracking_data.columns
            or "player_id" not in raw_tracking_data.columns
        ):
            from .tracking_loader import preprocess_tracking_data

            tracking_data = preprocess_tracking_data(raw_tracking_data)
        else:
            tracking_data = raw_tracking_data

        pressing_with_tracking = merge_pressing_with_tracking(
            pressing_with_phases,
            tracking_data,
            window_before=window_before,
            window_after=window_after,
        )

        result["pressing_with_tracking"] = pressing_with_tracking
        result["unified_dataset"] = pressing_with_tracking

    return result


def get_pressing_event_tracking_snapshot(
    tracking_data: pd.DataFrame,
    frame: int,
    player_ids: Optional[List[int]] = None,
    team_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Get a snapshot of tracking data for a specific frame, filtered by players/teams.

    Parameters
    ----------
    tracking_data : pd.DataFrame
        Preprocessed tracking data DataFrame
    frame : int
        Frame number
    player_ids : list of int, optional
        Specific player IDs to include. If None, includes all players.
    team_ids : list of int, optional
        Specific team IDs to include. If None, includes all teams.

    Returns
    -------
    pd.DataFrame
        Tracking snapshot for the specified frame
    """
    snapshot = tracking_data[tracking_data["frame"] == frame].copy()

    if player_ids is not None:
        snapshot = snapshot[snapshot["player_id"].isin(player_ids)]

    if team_ids is not None:
        snapshot = snapshot[snapshot["team_id"].isin(team_ids)]

    return snapshot


def aggregate_pressing_by_frame(pressing_with_tracking: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pressing event data by frame.

    Parameters
    ----------
    pressing_with_tracking : pd.DataFrame
        Merged pressing events and tracking data

    Returns
    -------
    pd.DataFrame
        Aggregated data by frame
    """
    # Group by frame and event
    frame_aggregated = (
        pressing_with_tracking.groupby(["frame", "event_id"])
        .agg(
            {
                "player_id": "count",  # Number of players in frame for this event
                "x": ["mean", "std", "min", "max"],
                "y": ["mean", "std", "min", "max"],
                "speed": ["mean", "std", "max"],
                "ball_x": "first",
                "ball_y": "first",
                "team_id": lambda x: x.nunique(),  # Number of teams
            }
        )
        .reset_index()
    )

    # Flatten column names
    frame_aggregated.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in frame_aggregated.columns.values
    ]

    return frame_aggregated


def get_pressing_event_summary(
    pressing_events: pd.DataFrame, tracking_data: Optional[pd.DataFrame] = None
) -> Dict[str, Union[int, float, pd.DataFrame]]:
    """
    Get summary statistics for pressing events.

    Parameters
    ----------
    pressing_events : pd.DataFrame
        Pressing events DataFrame
    tracking_data : pd.DataFrame, optional
        Preprocessed tracking data. If provided, includes tracking-based metrics.

    Returns
    -------
    dict
        Summary statistics dictionary
    """
    summary = {
        "total_pressing_events": len(pressing_events),
        "unique_players": (
            pressing_events["player_id"].nunique()
            if "player_id" in pressing_events.columns
            else 0
        ),
        "unique_teams": (
            pressing_events["team_id"].nunique()
            if "team_id" in pressing_events.columns
            else 0
        ),
    }

    # Add phase context summary
    if "phase_team_in_possession_phase_type" in pressing_events.columns:
        phase_summary = pressing_events[
            "phase_team_in_possession_phase_type"
        ].value_counts()
        summary["pressing_by_phase_type"] = phase_summary

    if "phase_team_out_of_possession_phase_type" in pressing_events.columns:
        defensive_phase_summary = pressing_events[
            "phase_team_out_of_possession_phase_type"
        ].value_counts()
        summary["pressing_by_defensive_phase_type"] = defensive_phase_summary

    # Add location summary
    if "third_start" in pressing_events.columns:
        location_summary = pressing_events["third_start"].value_counts()
        summary["pressing_by_location"] = location_summary

    return summary
