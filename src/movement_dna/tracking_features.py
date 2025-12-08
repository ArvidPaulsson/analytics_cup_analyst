"""
Tracking Feature Extraction for Movement DNA

Extracts cheap shape features from tracking data per possession or 1-second window:
- Distance to teammates
- Team width/depth when receiving ball
- Movement vector before receiving ball
- Separation gain (from events)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from scipy.spatial.distance import cdist

try:
    from kloppy import skillcorner

    KLOPPY_AVAILABLE = True
except ImportError:
    KLOPPY_AVAILABLE = False
    print("Warning: kloppy not available. Install with: pip install kloppy")

try:
    from databallpy import get_game_from_kloppy

    DATABALLPY_AVAILABLE = True
except ImportError:
    DATABALLPY_AVAILABLE = False
    print("Warning: databallpy not available. Install with: pip install databallpy")


def load_tracking_with_kloppy(match_id: int) -> pd.DataFrame:
    """
    Load tracking data using kloppy and convert to our format.

    Note: This is a simplified version. For production use, consider using
    the JSONL file approach which is more straightforward.

    Args:
        match_id: Match ID to load

    Returns:
        DataFrame with tracking data (columns: frame, player_id, x, y, team_id, match_id)
    """
    if not KLOPPY_AVAILABLE:
        raise ImportError("kloppy is required. Install with: pip install kloppy")

    try:
        dataset = skillcorner.load_open_data(
            match_id=match_id,
            coordinates="skillcorner",
        )

        # Use databallpy if available for easier conversion
        if DATABALLPY_AVAILABLE:
            try:
                game = get_game_from_kloppy(dataset)
                tracking_df = game.tracking_data

                # Databallpy provides a cleaner structure
                # Reshape to our format
                tracking_list = []
                for frame in tracking_df["frame"].unique():
                    frame_data = tracking_df[tracking_df["frame"] == frame]
                    for _, row in frame_data.iterrows():
                        # Databallpy structure varies - adapt as needed
                        # This is a placeholder - actual structure depends on databallpy version
                        pass

                # For now, recommend using JSONL approach
                raise NotImplementedError(
                    "Kloppy+databallpy integration needs databallpy structure knowledge. "
                    "Use extract_tracking_features_from_file() with JSONL files instead."
                )
            except Exception:
                pass

        # Fallback: recommend JSONL approach
        raise NotImplementedError(
            "Direct kloppy DataFrame reshaping is complex. "
            "Please use extract_tracking_features_from_file() with JSONL files, "
            "or implement kloppy reshaping based on your specific kloppy version."
        )

    except Exception as e:
        raise RuntimeError(f"Error loading tracking with kloppy: {e}")


def calculate_distance_to_teammates(
    player_pos: np.ndarray,
    teammate_positions: np.ndarray,
    n_nearest: int = 3,
) -> float:
    """
    Calculate average distance to nearest N teammates.

    Args:
        player_pos: Player position [x, y]
        teammate_positions: Array of teammate positions [[x1, y1], [x2, y2], ...]
        n_nearest: Number of nearest teammates to consider

    Returns:
        Average distance to nearest N teammates
    """
    if len(teammate_positions) == 0:
        return 0.0

    distances = np.sqrt(np.sum((teammate_positions - player_pos) ** 2, axis=1))

    n_consider = min(n_nearest, len(distances))
    nearest_distances = np.partition(distances, n_consider - 1)[:n_consider]

    return np.mean(nearest_distances)


def calculate_team_shape(team_positions: np.ndarray) -> Tuple[float, float]:
    """
    Calculate team width and depth.

    Args:
        team_positions: Array of team positions [[x1, y1], [x2, y2], ...]

    Returns:
        Tuple of (width, depth)
        - width: Standard deviation of y-coordinates
        - depth: max(x) - min(x)
    """
    if len(team_positions) == 0:
        return 0.0, 0.0

    x_coords = team_positions[:, 0]
    y_coords = team_positions[:, 1]

    width = np.std(y_coords) if len(y_coords) > 1 else 0.0
    depth = np.max(x_coords) - np.min(x_coords) if len(x_coords) > 0 else 0.0

    return width, depth


def calculate_movement_vector(
    tracking_df: pd.DataFrame,
    player_id: int,
    frame: int,
    frames_before: int = 10,  # 1 second at 10fps
) -> Tuple[float, float]:
    """
    Calculate movement vector (velocity) before a given frame.

    Args:
        tracking_df: Tracking DataFrame
        player_id: Player ID
        frame: Frame to calculate movement before
        frames_before: Number of frames to look back

    Returns:
        Tuple of (dx, dy) - movement vector
    """
    player_tracking = tracking_df[
        (tracking_df["player_id"] == player_id)
        & (tracking_df["frame"] <= frame)
        & (tracking_df["frame"] > frame - frames_before)
    ].sort_values("frame")

    if len(player_tracking) < 2:
        return 0.0, 0.0

    # Calculate velocity from position changes
    x_start = player_tracking["x"].iloc[0]
    y_start = player_tracking["y"].iloc[0]
    x_end = player_tracking["x"].iloc[-1]
    y_end = player_tracking["y"].iloc[-1]

    # Time difference (frames to seconds)
    time_diff = len(player_tracking) / 10.0  # 10 fps

    if time_diff == 0:
        return 0.0, 0.0

    dx = (x_end - x_start) / time_diff
    dy = (y_end - y_start) / time_diff

    return dx, dy


def extract_tracking_features(
    tracking_df: pd.DataFrame,
    events_df: pd.DataFrame,
    midfielder_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extract tracking features per player per match.

    Args:
        tracking_df: DataFrame with tracking data (must include match_id, frame, player_id, x, y, team_id)
        events_df: DataFrame with dynamic events (for pass receptions and separation_gain)
        midfielder_ids: Optional list of midfielder player IDs to filter

    Returns:
        DataFrame with columns: [player_id, match_id, avg_dist_to_teammates,
                                 avg_team_width, avg_team_depth,
                                 avg_movement_vector_x, avg_movement_vector_y,
                                 avg_separation_gain]
    """
    if tracking_df.empty or events_df.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "avg_dist_to_teammates",
                "avg_team_width",
                "avg_team_depth",
                "avg_movement_vector_x",
                "avg_movement_vector_y",
                "avg_separation_gain",
            ]
        )

    # Ensure required columns exist
    required_tracking_cols = ["match_id", "frame", "player_id", "x", "y"]
    missing_cols = [
        col for col in required_tracking_cols if col not in tracking_df.columns
    ]
    if missing_cols:
        raise ValueError(f"Tracking DataFrame missing required columns: {missing_cols}")

    # Get pass reception events
    # Pass receptions are events where end_type == "pass_reception" or
    # passing_option events where received == True
    pass_receptions = events_df[
        (
            (events_df["end_type"] == "pass_reception")
            | (events_df.get("received", pd.Series([False] * len(events_df))) == True)
        )
        & (events_df["event_type"] == "player_possession")
    ].copy()

    # Filter to midfielders if provided
    if midfielder_ids is not None:
        pass_receptions = pass_receptions[
            pass_receptions["player_id"].isin(midfielder_ids)
        ]

    if pass_receptions.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "avg_dist_to_teammates",
                "avg_team_width",
                "avg_team_depth",
                "avg_movement_vector_x",
                "avg_movement_vector_y",
                "avg_separation_gain",
            ]
        )

    # Get unique player-match combinations
    player_matches = pass_receptions[["player_id", "match_id"]].drop_duplicates()

    results = []

    for _, row in player_matches.iterrows():
        player_id = row["player_id"]
        match_id = row["match_id"]

        # Filter to this player-match
        player_match_receptions = pass_receptions[
            (pass_receptions["player_id"] == player_id)
            & (pass_receptions["match_id"] == match_id)
        ]

        player_match_tracking = tracking_df[
            (tracking_df["player_id"] == player_id)
            & (tracking_df["match_id"] == match_id)
        ]

        if player_match_tracking.empty:
            continue

        # Get team_id for this player (from events if not in tracking)
        if "team_id" in player_match_tracking.columns:
            team_id = (
                player_match_tracking["team_id"].iloc[0]
                if not player_match_tracking.empty
                else None
            )
        elif not events_df.empty and "team_id" in events_df.columns:
            player_events = events_df[
                (events_df["player_id"] == player_id)
                & (events_df["match_id"] == match_id)
            ]
            team_id = (
                player_events["team_id"].iloc[0] if not player_events.empty else None
            )
        else:
            team_id = None

        # Extract features for each pass reception
        dist_to_teammates_list = []
        team_width_list = []
        team_depth_list = []
        movement_vector_x_list = []
        movement_vector_y_list = []
        separation_gain_list = []

        for _, reception in player_match_receptions.iterrows():
            frame = reception.get("frame_end", reception.get("frame_start"))

            if pd.isna(frame):
                continue

            frame = int(frame)

            # Get tracking data at reception frame
            frame_tracking = tracking_df[
                (tracking_df["match_id"] == match_id) & (tracking_df["frame"] == frame)
            ]

            if frame_tracking.empty:
                continue

            # Get player position
            player_frame = frame_tracking[frame_tracking["player_id"] == player_id]
            if player_frame.empty:
                continue

            player_pos = np.array(
                [[player_frame["x"].iloc[0], player_frame["y"].iloc[0]]]
            )

            # Get teammate positions (same team, different player)
            if team_id is not None and "team_id" in frame_tracking.columns:
                teammates = frame_tracking[
                    (frame_tracking["team_id"] == team_id)
                    & (frame_tracking["player_id"] != player_id)
                    & (frame_tracking["x"].notna())
                    & (frame_tracking["y"].notna())
                ]
            else:
                # Fallback: use all other players (less accurate)
                teammates = frame_tracking[
                    (frame_tracking["player_id"] != player_id)
                    & (frame_tracking["x"].notna())
                    & (frame_tracking["y"].notna())
                ]

            if not teammates.empty:
                teammate_positions = teammates[["x", "y"]].values

                # Distance to teammates
                dist = calculate_distance_to_teammates(
                    player_pos[0], teammate_positions, n_nearest=3
                )
                dist_to_teammates_list.append(dist)

                # Team shape
                all_team_positions = np.vstack([player_pos, teammate_positions])
                width, depth = calculate_team_shape(all_team_positions)
                team_width_list.append(width)
                team_depth_list.append(depth)

            # Movement vector before receiving
            dx, dy = calculate_movement_vector(
                player_match_tracking, player_id, frame, frames_before=10
            )
            movement_vector_x_list.append(dx)
            movement_vector_y_list.append(dy)

            # Separation gain (from events)
            if "separation_gain" in reception.index and not pd.isna(
                reception["separation_gain"]
            ):
                separation_gain_list.append(reception["separation_gain"])

        # Calculate averages
        result = {
            "player_id": player_id,
            "match_id": match_id,
            "avg_dist_to_teammates": (
                np.mean(dist_to_teammates_list) if dist_to_teammates_list else 0.0
            ),
            "avg_team_width": np.mean(team_width_list) if team_width_list else 0.0,
            "avg_team_depth": np.mean(team_depth_list) if team_depth_list else 0.0,
            "avg_movement_vector_x": (
                np.mean(movement_vector_x_list) if movement_vector_x_list else 0.0
            ),
            "avg_movement_vector_y": (
                np.mean(movement_vector_y_list) if movement_vector_y_list else 0.0
            ),
            "avg_separation_gain": (
                np.mean(separation_gain_list) if separation_gain_list else 0.0
            ),
        }

        results.append(result)

    if not results:
        return pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "avg_dist_to_teammates",
                "avg_team_width",
                "avg_team_depth",
                "avg_movement_vector_x",
                "avg_movement_vector_y",
                "avg_separation_gain",
            ]
        )

    features_df = pd.DataFrame(results)

    # Ensure numeric columns are properly typed
    numeric_cols = [
        "avg_dist_to_teammates",
        "avg_team_width",
        "avg_team_depth",
        "avg_movement_vector_x",
        "avg_movement_vector_y",
        "avg_separation_gain",
    ]
    for col in numeric_cols:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(
                0
            )

    return features_df


def extract_tracking_features_from_kloppy(
    match_id: int,
    events_df: pd.DataFrame,
    midfielder_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Load tracking data using kloppy and extract features.

    Note: This function requires proper kloppy DataFrame reshaping.
    For now, it's recommended to use extract_tracking_features_from_file() with JSONL files.

    Args:
        match_id: Match ID
        events_df: DataFrame with dynamic events
        midfielder_ids: Optional list of midfielder player IDs to filter

    Returns:
        DataFrame with tracking features
    """
    # For now, raise NotImplementedError and recommend JSONL approach
    raise NotImplementedError(
        "Kloppy integration needs DataFrame reshaping implementation. "
        "Please use extract_tracking_features_from_file() with JSONL files instead."
    )

    # Future implementation would go here:
    # tracking_df = load_tracking_with_kloppy(match_id)
    # return extract_tracking_features(tracking_df, events_df, midfielder_ids)


def extract_tracking_features_from_file(
    tracking_file: Path,
    events_df: pd.DataFrame,
    midfielder_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Load tracking data from JSONL file and extract features.

    Args:
        tracking_file: Path to tracking JSONL file
        events_df: DataFrame with dynamic events
        midfielder_ids: Optional list of midfielder player IDs to filter

    Returns:
        DataFrame with tracking features
    """
    if not tracking_file.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracking_file}")

    # Load JSONL
    tracking_raw = pd.read_json(tracking_file, lines=True)

    # Extract match_id from filename
    match_id = int(tracking_file.stem.split("_")[0])

    # Flatten player_data
    tracking_df = pd.json_normalize(
        tracking_raw.to_dict("records"),
        "player_data",
        ["frame", "timestamp", "period", "possession", "ball_data"],
    )

    # Extract possession info
    tracking_df["possession_player_id"] = tracking_df["possession"].apply(
        lambda x: x.get("player_id") if isinstance(x, dict) else None
    )
    tracking_df["possession_group"] = tracking_df["possession"].apply(
        lambda x: x.get("group") if isinstance(x, dict) else None
    )

    tracking_df["match_id"] = match_id

    # Get team_id from events (match player_id to team_id)
    if not events_df.empty and "team_id" in events_df.columns:
        player_teams = events_df[["player_id", "team_id"]].drop_duplicates()
        tracking_df = tracking_df.merge(player_teams, on="player_id", how="left")
    else:
        # Fallback: try to infer from possession_group
        # This is less reliable but better than nothing
        tracking_df["team_id"] = tracking_df["possession_group"].apply(
            lambda x: 1 if x == "home" else 2 if x == "away" else None
        )

    return extract_tracking_features(tracking_df, events_df, midfielder_ids)
