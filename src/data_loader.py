"""
Data loader for phases of play data from SkillCorner Open Data.
Loads data directly from GitHub and preprocesses for transition analysis.
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Tuple
import requests
from io import StringIO


# GitHub base URL for SkillCorner Open Data
GITHUB_BASE_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"


def load_matches_metadata() -> pd.DataFrame:
    """
    Load matches metadata from GitHub.

    Returns:
        DataFrame with match information including IDs, teams, dates, etc.
    """
    url = f"{GITHUB_BASE_URL}/matches.json"
    response = requests.get(url)
    response.raise_for_status()
    matches_data = json.loads(response.text)
    return pd.DataFrame(matches_data)


def load_phases_of_play(match_id: int) -> pd.DataFrame:
    """
    Load phases of play data for a specific match from GitHub.

    Args:
        match_id: Match ID to load data for

    Returns:
        DataFrame with phases of play data
    """
    url = f"{GITHUB_BASE_URL}/matches/{match_id}/{match_id}_phases_of_play.csv"
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    return df


def load_match_metadata(match_id: int) -> Dict:
    """
    Load match metadata (lineups, scores, etc.) for a specific match.

    Args:
        match_id: Match ID to load data for

    Returns:
        Dictionary with match metadata
    """
    url = f"{GITHUB_BASE_URL}/matches/{match_id}/{match_id}_match.json"
    response = requests.get(url)
    response.raise_for_status()
    return json.loads(response.text)


def preprocess_phases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess phases of play data for transition analysis.

    Adds computed columns and ensures data consistency.

    Args:
        df: Raw phases of play DataFrame

    Returns:
        Preprocessed DataFrame with additional columns
    """
    df = df.copy()

    # Ensure phase types are strings (handle any NaN values)
    df["team_in_possession_phase_type"] = df["team_in_possession_phase_type"].fillna(
        "unknown"
    )
    df["team_out_of_possession_phase_type"] = df[
        "team_out_of_possession_phase_type"
    ].fillna("unknown")

    # Create a combined phase identifier for easier analysis
    df["phase_combination"] = (
        df["team_in_possession_phase_type"]
        + "_vs_"
        + df["team_out_of_possession_phase_type"]
    )

    # Add outcome flags
    df["leads_to_goal"] = df["team_possession_lead_to_goal"].fillna(False)
    df["leads_to_shot"] = df["team_possession_lead_to_shot"].fillna(False)
    df["possession_loss"] = df["team_possession_loss_in_phase"].fillna(False)

    # Create outcome category
    df["outcome"] = "no_outcome"
    df.loc[df["leads_to_goal"], "outcome"] = "goal"
    df.loc[df["leads_to_shot"] & ~df["leads_to_goal"], "outcome"] = "shot"

    # Add minute as float for easier filtering
    df["minute_float"] = df["minute_start"] + (df["second_start"] / 60.0)

    # Ensure spatial data is numeric
    spatial_cols = ["x_start", "y_start", "x_end", "y_end"]
    for col in spatial_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by match_id, period, and frame_start for proper sequence analysis
    df = df.sort_values(["match_id", "period", "frame_start"]).reset_index(drop=True)

    return df


def identify_possession_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify possession changes and create possession sequences.

    Args:
        df: Preprocessed phases DataFrame

    Returns:
        DataFrame with possession change indicators and sequences
    """
    df = df.copy()

    # Mark possession changes
    df["possession_change"] = False

    # Possession changes occur when:
    # 1. team_possession_loss_in_phase is True, OR
    # 2. team_in_possession_id changes from previous row
    df["prev_team_in_possession"] = df["team_in_possession_id"].shift(1)
    df["prev_match_id"] = df["match_id"].shift(1)
    df["prev_period"] = df["period"].shift(1)

    # Only consider changes within the same match and period
    same_match_period = (df["match_id"] == df["prev_match_id"]) & (
        df["period"] == df["prev_period"]
    )

    team_change = df["team_in_possession_id"] != df["prev_team_in_possession"]
    df.loc[same_match_period & team_change, "possession_change"] = True
    df.loc[df["possession_loss"], "possession_change"] = True

    # Create possession sequence IDs
    df["possession_sequence"] = (
        df.groupby(["match_id", "period"])["possession_change"]
        .cumsum()
        .fillna(0)
        .astype(int)
    )

    # Create phase sequence within each possession
    df["phase_sequence"] = (
        df.groupby(["match_id", "period", "possession_sequence"]).cumcount() + 1
    )

    # Drop helper columns
    df = df.drop(columns=["prev_team_in_possession", "prev_match_id", "prev_period"])

    return df


def identify_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify transitions between phases within possessions.

    Args:
        df: DataFrame with possession sequences

    Returns:
        DataFrame with transition information
    """
    df = df.copy()

    # Get previous phase type within the same possession
    df["prev_phase_type"] = df.groupby(["match_id", "period", "possession_sequence"])[
        "team_in_possession_phase_type"
    ].shift(1)

    df["prev_outcome"] = df.groupby(["match_id", "period", "possession_sequence"])[
        "outcome"
    ].shift(1)

    # Create transition labels
    # Transitions occur when there's a previous phase type within the same possession
    # Since prev_phase_type is already grouped by possession_sequence, we just need to check if it exists
    df["transition"] = None
    has_prev_phase = df["prev_phase_type"].notna()

    df.loc[has_prev_phase, "transition"] = (
        df.loc[has_prev_phase, "prev_phase_type"]
        + " -> "
        + df.loc[has_prev_phase, "team_in_possession_phase_type"]
    )

    # Mark transitions to goals/shots
    df["transition_leads_to_goal"] = (df["transition"].notna()) & (df["leads_to_goal"])
    df["transition_leads_to_shot"] = (df["transition"].notna()) & (df["leads_to_shot"])

    return df


def load_and_preprocess_match(match_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and preprocess all data for a single match.

    Args:
        match_id: Match ID to load

    Returns:
        Tuple of (phases DataFrame, match metadata dictionary)
    """
    # Load phases of play
    phases_df = load_phases_of_play(match_id)

    # Preprocess
    phases_df = preprocess_phases(phases_df)
    phases_df = identify_possession_changes(phases_df)
    phases_df = identify_transitions(phases_df)

    # Load match metadata
    match_metadata = load_match_metadata(match_id)

    return phases_df, match_metadata


def load_multiple_matches(match_ids: List[int]) -> pd.DataFrame:
    """
    Load and combine phases of play data for multiple matches.

    Args:
        match_ids: List of match IDs to load

    Returns:
        Combined DataFrame with phases from all matches
    """
    all_phases = []

    for match_id in match_ids:
        try:
            phases_df, _ = load_and_preprocess_match(match_id)
            all_phases.append(phases_df)
        except Exception as e:
            print(f"Error loading match {match_id}: {e}")
            continue

    if not all_phases:
        return pd.DataFrame()

    return pd.concat(all_phases, ignore_index=True)
