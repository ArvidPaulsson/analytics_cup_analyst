"""
Tracking data loading and preprocessing utilities.

This module provides functions to load and preprocess tracking data,
including frame extraction and player position data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, cast
import pandas as pd
import numpy as np
from .data_loader import (
    load_tracking_data,
    load_tracking_dataset_kloppy,
    KLOPPY_AVAILABLE,
)


def _is_kloppy_dataset(obj: object) -> bool:
    """Heuristic check to detect Kloppy tracking datasets."""
    return KLOPPY_AVAILABLE and hasattr(obj, "to_df") and hasattr(obj, "transform")


def preprocess_tracking_data(
    raw_tracking: Union[pd.DataFrame, object],
    *,
    engine: str = "pandas",
    orientation: Optional[str] = None,
) -> pd.DataFrame:
    """
    Preprocess raw tracking data into a normalized format.

    Parameters
    ----------
    raw_tracking : pd.DataFrame or Kloppy TrackingDataset
        Raw tracking data from JSONL file or Kloppy dataset

    Returns
    -------
    pd.DataFrame
        Preprocessed tracking data with player positions normalized
    """
    if not _is_kloppy_dataset(raw_tracking):
        raise TypeError(
            "preprocess_tracking_data expects a Kloppy TrackingDataset. "
            "Ensure you load tracking data via Kloppy utilities."
        )

    dataset = raw_tracking
    if orientation:
        dataset = dataset.transform(to_orientation=orientation)

    df = dataset.to_df(engine=engine)  # type: ignore[attr-defined]
    if engine == "polars":
        df = df.to_pandas()  # type: ignore[call-arg]
    tracking_df = cast(pd.DataFrame, df)

    # Harmonise column names regardless of source
    rename_candidates = {
        "frame_id": "frame",
        "frame.id": "frame",
        "frame_timestamp": "timestamp",
        "frame.timestamp": "timestamp",
        "player_id": "player_id",
        "player.id": "player_id",
        "team_id": "team_id",
        "player.team_id": "team_id",
        "player.team.id": "team_id",
        "coordinates.x": "x",
        "coordinates.y": "y",
        "coordinates.z": "z",
        "player_coordinates.x": "x",
        "player_coordinates.y": "y",
        "player_coordinates.z": "z",
        "velocity": "speed",
        "ball_x": "ball_x",
        "ball_y": "ball_y",
        "ball_z": "ball_z",
        "ball.coordinates.x": "ball_x",
        "ball.coordinates.y": "ball_y",
        "ball.coordinates.z": "ball_z",
    }
    rename_map = {
        src: dst for src, dst in rename_candidates.items() if src in tracking_df.columns
    }
    tracking_df = tracking_df.rename(columns=rename_map)

    # Ensure essential columns exist (fill with default None if missing)
    required_columns = [
        "frame",
        "timestamp",
        "player_id",
        "team_id",
        "x",
        "y",
        "z",
        "speed",
        "ball_x",
        "ball_y",
        "ball_z",
    ]
    for col in required_columns:
        if col not in tracking_df.columns:
            tracking_df[col] = None

    return tracking_df


def extract_frame_data(tracking_data: pd.DataFrame, frame: int) -> pd.DataFrame:
    """
    Extract tracking data for a specific frame.

    Parameters
    ----------
    tracking_data : pd.DataFrame
        Preprocessed tracking data
    frame : int
        Frame number to extract

    Returns
    -------
    pd.DataFrame
        Tracking data for the specified frame
    """
    return tracking_data[tracking_data["frame"] == frame].copy()


def extract_frame_range(
    tracking_data: pd.DataFrame, frame_start: int, frame_end: int
) -> pd.DataFrame:
    """
    Extract tracking data for a range of frames.

    Parameters
    ----------
    tracking_data : pd.DataFrame
        Preprocessed tracking data
    frame_start : int
        Start frame
    frame_end : int
        End frame (inclusive)

    Returns
    -------
    pd.DataFrame
        Tracking data for the specified frame range
    """
    return tracking_data[
        (tracking_data["frame"] >= frame_start) & (tracking_data["frame"] <= frame_end)
    ].copy()


def get_player_positions(
    tracking_data: pd.DataFrame, frame: int, player_ids: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Get player positions for a specific frame.

    Parameters
    ----------
    tracking_data : pd.DataFrame
        Preprocessed tracking data
    frame : int
        Frame number
    player_ids : list of int, optional
        Specific player IDs to filter. If None, returns all players.

    Returns
    -------
    pd.DataFrame
        Player positions for the specified frame
    """
    frame_data = extract_frame_data(tracking_data, frame)

    if player_ids is not None:
        frame_data = frame_data[frame_data["player_id"].isin(player_ids)]

    return frame_data[["player_id", "team_id", "x", "y", "z", "speed"]].copy()


def get_ball_position(
    tracking_data: pd.DataFrame, frame: int
) -> Optional[Dict[str, float]]:
    """
    Get ball position for a specific frame.

    Parameters
    ----------
    tracking_data : pd.DataFrame
        Preprocessed tracking data
    frame : int
        Frame number

    Returns
    -------
    dict or None
        Ball position dictionary with 'x', 'y', 'z' keys, or None if not found
    """
    frame_data = extract_frame_data(tracking_data, frame)

    if len(frame_data) == 0:
        return None

    # Ball position should be the same for all players in a frame
    first_row = frame_data.iloc[0]

    ball_pos = {
        "x": first_row.get("ball_x"),
        "y": first_row.get("ball_y"),
        "z": first_row.get("ball_z"),
    }

    # Return None if all values are None
    if all(v is None for v in ball_pos.values()):
        return None

    return ball_pos


def get_team_positions(
    tracking_data: pd.DataFrame, frame: int, team_id: int
) -> pd.DataFrame:
    """
    Get all player positions for a specific team in a frame.

    Parameters
    ----------
    tracking_data : pd.DataFrame
        Preprocessed tracking data
    frame : int
        Frame number
    team_id : int
        Team ID

    Returns
    -------
    pd.DataFrame
        Player positions for the specified team
    """
    frame_data = extract_frame_data(tracking_data, frame)
    team_data = frame_data[frame_data["team_id"] == team_id].copy()

    return team_data[["player_id", "x", "y", "z", "speed"]].copy()


def calculate_player_distances(
    tracking_data: pd.DataFrame, frame: int, reference_player_id: int
) -> pd.DataFrame:
    """
    Calculate distances from a reference player to all other players in a frame.

    Parameters
    ----------
    tracking_data : pd.DataFrame
        Preprocessed tracking data
    frame : int
        Frame number
    reference_player_id : int
        Player ID to use as reference

    Returns
    -------
    pd.DataFrame
        DataFrame with distances from reference player
    """
    frame_data = extract_frame_data(tracking_data, frame)

    # Get reference player position
    ref_player = frame_data[frame_data["player_id"] == reference_player_id]

    if len(ref_player) == 0:
        return pd.DataFrame()

    ref_x = ref_player.iloc[0]["x"]
    ref_y = ref_player.iloc[0]["y"]

    # Calculate distances
    distances = []
    for _, player in frame_data.iterrows():
        if player["player_id"] != reference_player_id:
            dx = player["x"] - ref_x
            dy = player["y"] - ref_y
            distance = np.sqrt(dx**2 + dy**2)

            distances.append(
                {
                    "player_id": player["player_id"],
                    "team_id": player["team_id"],
                    "x": player["x"],
                    "y": player["y"],
                    "distance": distance,
                }
            )

    return pd.DataFrame(distances)


def load_and_preprocess_tracking(
    match_id: int,
    data_dir: Optional[Union[str, Path]] = None,
    use_github: bool = True,
    *,
    engine: str = "pandas",
    orientation: Optional[str] = "STATIC_HOME_AWAY",
    sample_rate: Optional[float] = None,
    limit: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load and preprocess tracking data for a match.

    Parameters
    ----------
    match_id : int
        Match ID
    data_dir : str or Path, optional
        Local directory containing match data
    use_github : bool, default False
        If True, load from GitHub. If False, load from local files.

    Returns
    -------
    pd.DataFrame
        Preprocessed tracking data
    """
    dataset = load_tracking_dataset_kloppy(
        match_id,
        coordinates=kwargs.pop("coordinates", "skillcorner"),
        sample_rate=sample_rate,
        limit=limit,
        to_orientation=orientation,
        use_github=use_github,
        data_dir=data_dir,
        **kwargs,
    )
    return preprocess_tracking_data(
        dataset,
        engine=engine,
        orientation=orientation,
    )


def get_tracking_for_event(
    tracking_data: pd.DataFrame,
    frame_start: int,
    frame_end: int,
    window_before: int = 0,
    window_after: int = 0,
) -> pd.DataFrame:
    """
    Get tracking data for an event with optional time windows.

    Parameters
    ----------
    tracking_data : pd.DataFrame
        Preprocessed tracking data
    frame_start : int
        Event start frame
    frame_end : int
        Event end frame
    window_before : int, default 0
        Number of frames before event start to include
    window_after : int, default 0
        Number of frames after event end to include

    Returns
    -------
    pd.DataFrame
        Tracking data for the event window
    """
    actual_start = frame_start - window_before
    actual_end = frame_end + window_after

    return extract_frame_range(tracking_data, actual_start, actual_end)
