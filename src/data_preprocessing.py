"""
Data preprocessing utilities for pressing analysis.

This module provides functions for coordinate normalization, missing data handling,
and derived metrics calculation.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


# Standard pitch dimensions (in meters)
STANDARD_PITCH_LENGTH = 105.0
STANDARD_PITCH_WIDTH = 68.0


def normalize_coordinates(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    pitch_length: float = STANDARD_PITCH_LENGTH,
    pitch_width: float = STANDARD_PITCH_WIDTH,
    normalize_to_0_1: bool = False,
) -> pd.DataFrame:
    """
    Normalize coordinates to standard pitch dimensions or 0-1 range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with coordinate columns
    x_col : str, default 'x'
        Name of x-coordinate column
    y_col : str, default 'y'
        Name of y-coordinate column
    pitch_length : float, default 105.0
        Standard pitch length in meters
    pitch_width : float, default 68.0
        Standard pitch width in meters
    normalize_to_0_1 : bool, default False
        If True, normalize to 0-1 range. If False, normalize to standard pitch dimensions.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized coordinates
    """
    df = df.copy()

    if x_col not in df.columns or y_col not in df.columns:
        return df

    # Get actual pitch dimensions from data
    x_min = df[x_col].min()
    x_max = df[x_col].max()
    y_min = df[y_col].min()
    y_max = df[y_col].max()

    # Calculate scale factors
    x_range = x_max - x_min
    y_range = y_max - y_min

    if normalize_to_0_1:
        # Normalize to 0-1 range
        df[f"{x_col}_normalized"] = (df[x_col] - x_min) / x_range if x_range > 0 else 0
        df[f"{y_col}_normalized"] = (df[y_col] - y_min) / y_range if y_range > 0 else 0
    else:
        # Normalize to standard pitch dimensions
        # Assuming data is centered at origin, scale to fit standard dimensions
        x_scale = pitch_length / x_range if x_range > 0 else 1
        y_scale = pitch_width / y_range if y_range > 0 else 1

        df[f"{x_col}_normalized"] = df[x_col] * x_scale
        df[f"{y_col}_normalized"] = df[y_col] * y_scale

    return df


def handle_missing_data(
    df: pd.DataFrame,
    strategy: str = "forward_fill",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Handle missing data in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential missing data
    strategy : str, default 'forward_fill'
        Strategy for handling missing data:
        - 'forward_fill': Forward fill missing values
        - 'backward_fill': Backward fill missing values
        - 'interpolate': Linear interpolation
        - 'drop': Drop rows with missing values
        - 'zero': Fill with zeros
        - 'mean': Fill with mean value
    columns : list of str, optional
        Specific columns to process. If None, processes all columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing data handled
    """
    df = df.copy()

    if columns is None:
        columns = df.columns.tolist()
    else:
        columns = [col for col in columns if col in df.columns]

    for col in columns:
        if df[col].isna().any():
            if strategy == "forward_fill":
                df[col] = df[col].fillna(method="ffill")
            elif strategy == "backward_fill":
                df[col] = df[col].fillna(method="bfill")
            elif strategy == "interpolate":
                df[col] = df[col].interpolate()
            elif strategy == "drop":
                df = df.dropna(subset=[col])
            elif strategy == "zero":
                df[col] = df[col].fillna(0)
            elif strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())

    return df


def calculate_distance(
    x1: Union[float, pd.Series],
    y1: Union[float, pd.Series],
    x2: Union[float, pd.Series],
    y2: Union[float, pd.Series],
) -> Union[float, pd.Series]:
    """
    Calculate Euclidean distance between two points.

    Parameters
    ----------
    x1 : float or pd.Series
        X-coordinate of first point
    y1 : float or pd.Series
        Y-coordinate of first point
    x2 : float or pd.Series
        X-coordinate of second point
    y2 : float or pd.Series
        Y-coordinate of second point

    Returns
    -------
    float or pd.Series
        Distance between points
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_angle(
    x1: Union[float, pd.Series],
    y1: Union[float, pd.Series],
    x2: Union[float, pd.Series],
    y2: Union[float, pd.Series],
) -> Union[float, pd.Series]:
    """
    Calculate angle (in degrees) from point 1 to point 2.

    Parameters
    ----------
    x1 : float or pd.Series
        X-coordinate of first point
    y1 : float or pd.Series
        Y-coordinate of first point
    x2 : float or pd.Series
        X-coordinate of second point
    y2 : float or pd.Series
        Y-coordinate of second point

    Returns
    -------
    float or pd.Series
        Angle in degrees (0-360)
    """
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(dy, dx))
    # Normalize to 0-360 range
    angle = angle % 360
    return angle


def calculate_speed(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    time_col: str = "timestamp",
    frame_col: str = "frame",
    player_id_col: str = "player_id",
) -> pd.DataFrame:
    """
    Calculate player speed from position data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with position data
    x_col : str, default 'x'
        Name of x-coordinate column
    y_col : str, default 'y'
        Name of y-coordinate column
    time_col : str, default 'timestamp'
        Name of timestamp column (if available)
    frame_col : str, default 'frame'
        Name of frame column
    player_id_col : str, default 'player_id'
        Name of player ID column

    Returns
    -------
    pd.DataFrame
        DataFrame with speed calculated
    """
    df = df.copy()
    df = df.sort_values([player_id_col, frame_col])

    # Calculate displacement
    df["dx"] = df.groupby(player_id_col)[x_col].diff()
    df["dy"] = df.groupby(player_id_col)[y_col].diff()

    # Calculate distance
    df["distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    # Calculate speed (distance per frame or per second)
    if time_col in df.columns:
        df["dt"] = df.groupby(player_id_col)[time_col].diff()
        df["speed"] = df["distance"] / df["dt"]
        df["speed"] = df["speed"].fillna(0)
        # Clean up temporary columns including dt
        df = df.drop(columns=["dx", "dy", "distance", "dt"], errors="ignore")
    else:
        # Assume 25 fps (typical for tracking data)
        fps = 25
        df["speed"] = df["distance"] * fps
        # Clean up temporary columns
        df = df.drop(columns=["dx", "dy", "distance"], errors="ignore")

    return df


def calculate_derived_metrics(
    df: pd.DataFrame,
    pressing_player_col: str = "player_id",
    pressed_player_col: str = "player_in_possession_id",
    ball_x_col: str = "ball_x",
    ball_y_col: str = "ball_y",
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """
    Calculate derived metrics for pressing analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with pressing event and tracking data
    pressing_player_col : str, default 'player_id'
        Name of column with pressing player ID
    pressed_player_col : str, default 'player_in_possession_id'
        Name of column with pressed player ID
    ball_x_col : str, default 'ball_x'
        Name of ball x-coordinate column
    ball_y_col : str, default 'ball_y'
        Name of ball y-coordinate column
    x_col : str, default 'x'
        Name of player x-coordinate column
    y_col : str, default 'y'
        Name of player y-coordinate column

    Returns
    -------
    pd.DataFrame
        DataFrame with derived metrics added
    """
    df = df.copy()

    # Calculate distance from pressing player to ball
    if all(col in df.columns for col in [x_col, y_col, ball_x_col, ball_y_col]):
        df["distance_to_ball"] = calculate_distance(
            df[x_col], df[y_col], df[ball_x_col], df[ball_y_col]
        )

    # Calculate distance from pressing player to pressed player
    if pressing_player_col in df.columns and pressed_player_col in df.columns:
        pressing_players = df[df[pressing_player_col].notna()]
        pressed_players = df[df[pressed_player_col].notna()]

        # This would require merging or grouping - simplified version
        # For full implementation, would need to merge on frame and player IDs

    # Calculate angle to goal (assuming goal is at x=52.5, y=0 for standard pitch)
    goal_x = 52.5
    goal_y = 0
    if all(col in df.columns for col in [x_col, y_col]):
        df["angle_to_goal"] = calculate_angle(df[x_col], df[y_col], goal_x, goal_y)
        df["distance_to_goal"] = calculate_distance(
            df[x_col], df[y_col], goal_x, goal_y
        )

    return df


def add_pitch_zones(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    pitch_length: float = STANDARD_PITCH_LENGTH,
    pitch_width: float = STANDARD_PITCH_WIDTH,
    n_zones_x: int = 3,
    n_zones_y: int = 3,
) -> pd.DataFrame:
    """
    Add pitch zone labels to DataFrame based on coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with coordinate columns
    x_col : str, default 'x'
        Name of x-coordinate column
    y_col : str, default 'y'
        Name of y-coordinate column
    pitch_length : float, default 105.0
        Pitch length in meters
    pitch_width : float, default 68.0
        Pitch width in meters
    n_zones_x : int, default 3
        Number of zones in x-direction (length)
    n_zones_y : int, default 3
        Number of zones in y-direction (width)

    Returns
    -------
    pd.DataFrame
        DataFrame with pitch zone labels added
    """
    df = df.copy()

    if x_col not in df.columns or y_col not in df.columns:
        return df

    # Calculate zone boundaries
    x_min = df[x_col].min()
    x_max = df[x_col].max()
    y_min = df[y_col].min()
    y_max = df[y_col].max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Create zone labels
    df["zone_x"] = pd.cut(
        df[x_col],
        bins=n_zones_x,
        labels=[f"zone_x_{i+1}" for i in range(n_zones_x)],
        include_lowest=True,
    )

    df["zone_y"] = pd.cut(
        df[y_col],
        bins=n_zones_y,
        labels=[f"zone_y_{i+1}" for i in range(n_zones_y)],
        include_lowest=True,
    )

    df["zone"] = df["zone_x"].astype(str) + "_" + df["zone_y"].astype(str)

    return df


def preprocess_pressing_data(
    df: pd.DataFrame,
    normalize_coords: bool = True,
    handle_missing: bool = True,
    calculate_metrics: bool = True,
    add_zones: bool = True,
) -> pd.DataFrame:
    """
    Comprehensive preprocessing pipeline for pressing data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with pressing event and tracking data
    normalize_coords : bool, default True
        If True, normalize coordinates
    handle_missing : bool, default True
        If True, handle missing data
    calculate_metrics : bool, default True
        If True, calculate derived metrics
    add_zones : bool, default True
        If True, add pitch zone labels

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame
    """
    df = df.copy()

    if handle_missing:
        df = handle_missing_data(df, strategy="forward_fill")

    if normalize_coords:
        df = normalize_coordinates(df)

    if calculate_metrics:
        df = calculate_derived_metrics(df)

    if add_zones:
        df = add_pitch_zones(df)

    return df


def get_pressing_intensity(
    df: pd.DataFrame, x_col: str = "x", y_col: str = "y", bin_size: float = 5.0
) -> pd.DataFrame:
    """
    Calculate pressing intensity heatmap data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with pressing event coordinates
    x_col : str, default 'x'
        Name of x-coordinate column
    y_col : str, default 'y'
        Name of y-coordinate column
    bin_size : float, default 5.0
        Size of bins for heatmap in meters

    Returns
    -------
    pd.DataFrame
        Heatmap data with intensity values
    """
    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame()

    # Create bins
    x_min = df[x_col].min()
    x_max = df[x_col].max()
    y_min = df[y_col].min()
    y_max = df[y_col].max()

    x_bins = np.arange(x_min, x_max + bin_size, bin_size)
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)

    # Calculate 2D histogram
    intensity, x_edges, y_edges = np.histogram2d(
        df[x_col], df[y_col], bins=[x_bins, y_bins]
    )

    # Create DataFrame with intensity values
    intensity_df = pd.DataFrame(intensity.T, columns=x_edges[:-1], index=y_edges[:-1])
    intensity_df = intensity_df.stack().reset_index()
    intensity_df.columns = [y_col, x_col, "intensity"]
    intensity_df[x_col] = intensity_df[x_col].astype(float)
    intensity_df[y_col] = intensity_df[y_col].astype(float)

    return intensity_df
