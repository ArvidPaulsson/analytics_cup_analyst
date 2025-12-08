"""
Physical Feature Extraction for Movement DNA

Extracts season-level physical features from aggregates CSV for player clustering.
These features are used to initially cluster players based on physical similarities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


def extract_physical_features(
    physical_aggregates_df: pd.DataFrame,
    filter_midfielders: bool = True,
) -> pd.DataFrame:
    """
    Extract physical features from season-level aggregates for clustering.

    Args:
        physical_aggregates_df: DataFrame with physical aggregates (from CSV)
        filter_midfielders: If True, filter to position_group == "Midfield"

    Returns:
        DataFrame with columns: [player_id, sprint_count, meters_per_min,
                                 explosive_actions, sprint_distance, hsr_bursts]
    """
    if physical_aggregates_df.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "sprint_count",
                "meters_per_min",
                "explosive_actions",
                "sprint_distance",
                "hsr_bursts",
            ]
        )

    # Filter to midfielders if requested
    if filter_midfielders:
        df = physical_aggregates_df[
            physical_aggregates_df["position_group"] == "Midfield"
        ].copy()
    else:
        df = physical_aggregates_df.copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "sprint_count",
                "meters_per_min",
                "explosive_actions",
                "sprint_distance",
                "hsr_bursts",
            ]
        )

    # Aggregate by player_id (in case there are multiple rows per player)
    # Take mean of physical metrics
    agg_dict = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if col != "player_id":
            agg_dict[col] = "mean"

    # Keep first value for non-numeric columns
    for col in df.columns:
        if col not in numeric_cols and col != "player_id":
            agg_dict[col] = "first"

    df_agg = df.groupby("player_id").agg(agg_dict).reset_index()

    # Extract features
    features = pd.DataFrame({"player_id": df_agg["player_id"]})

    # 1. Sprint count
    if "sprint_count_full_all" in df_agg.columns:
        features["sprint_count"] = df_agg["sprint_count_full_all"].fillna(0)
    else:
        features["sprint_count"] = 0

    # 2. Meters per minute
    if "total_metersperminute_full_all" in df_agg.columns:
        features["meters_per_min"] = df_agg["total_metersperminute_full_all"].fillna(0)
    else:
        features["meters_per_min"] = 0

    # 3. Explosive actions
    # Sum of explacceltohsr_count and explacceltosprint_count
    explosive_actions = 0
    if "explacceltohsr_count_full_all" in df_agg.columns:
        explosive_actions += df_agg["explacceltohsr_count_full_all"].fillna(0)
    if "explacceltosprint_count_full_all" in df_agg.columns:
        explosive_actions += df_agg["explacceltosprint_count_full_all"].fillna(0)
    features["explosive_actions"] = explosive_actions

    # 4. Sprint distance
    if "sprint_distance_full_all" in df_agg.columns:
        features["sprint_distance"] = df_agg["sprint_distance_full_all"].fillna(0)
    else:
        features["sprint_distance"] = 0

    # 5. HSR bursts
    # Use HSR count as proxy for HSR bursts (consecutive HSR sequences)
    # Alternatively, could calculate from tracking data, but for season-level clustering,
    # HSR count is a reasonable proxy
    if "hsr_count_full_all" in df_agg.columns:
        features["hsr_bursts"] = df_agg["hsr_count_full_all"].fillna(0)
    else:
        features["hsr_bursts"] = 0

    # Ensure all numeric columns are properly typed
    numeric_cols = [
        "sprint_count",
        "meters_per_min",
        "explosive_actions",
        "sprint_distance",
        "hsr_bursts",
    ]
    for col in numeric_cols:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)

    return features


def load_and_extract_physical_features(
    aggregates_file: Path,
    filter_midfielders: bool = True,
) -> pd.DataFrame:
    """
    Load physical aggregates CSV and extract features.

    Args:
        aggregates_file: Path to physical aggregates CSV file
        filter_midfielders: If True, filter to position_group == "Midfield"

    Returns:
        DataFrame with physical features
    """
    if not aggregates_file.exists():
        raise FileNotFoundError(
            f"Physical aggregates file not found: {aggregates_file}"
        )

    physical_df = pd.read_csv(aggregates_file)

    return extract_physical_features(physical_df, filter_midfielders)


def get_physical_features_for_clustering(
    physical_aggregates_df: pd.DataFrame,
    player_ids: Optional[List[int]] = None,
    filter_midfielders: bool = True,
) -> pd.DataFrame:
    """
    Extract physical features for specific players (useful for clustering subset).

    Args:
        physical_aggregates_df: DataFrame with physical aggregates
        player_ids: Optional list of player IDs to filter. If None, includes all.
        filter_midfielders: If True, filter to position_group == "Midfield"

    Returns:
        DataFrame with physical features for specified players
    """
    features = extract_physical_features(physical_aggregates_df, filter_midfielders)

    if player_ids is not None:
        features = features[features["player_id"].isin(player_ids)]

    return features
