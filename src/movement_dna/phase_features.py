"""
Phase Context Feature Extraction for Movement DNA

Extracts phase-contextualized features from phases of play data.
This adds tactical context to movement patterns by analyzing player behavior
within different phases (create, build_up, finish, transition, etc.).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict


def get_phase_for_frame(
    phases_df: pd.DataFrame, frame: int, match_id: int
) -> Optional[Dict]:
    """
    Get phase information for a specific frame.

    Args:
        phases_df: DataFrame with phases of play
        frame: Frame number
        match_id: Match ID

    Returns:
        Dictionary with phase information or None if no phase found
    """
    phase = phases_df[
        (phases_df["match_id"] == match_id)
        & (phases_df["frame_start"] <= frame)
        & (phases_df["frame_end"] >= frame)
    ]

    if phase.empty:
        return None

    # Return first matching phase (should be unique)
    return phase.iloc[0].to_dict()


def extract_phase_context_features(
    phases_df: pd.DataFrame,
    events_df: pd.DataFrame,
    tracking_df: Optional[pd.DataFrame] = None,
    midfielder_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extract phase-contextualized features per player per match.

    Args:
        phases_df: DataFrame with phases of play
        events_df: DataFrame with dynamic events
        tracking_df: Optional DataFrame with tracking data
        midfielder_ids: Optional list of midfielder player IDs to filter

    Returns:
        DataFrame with phase-contextualized features per player per match
    """
    if phases_df.empty or events_df.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "phase_distribution_create",
                "phase_distribution_build_up",
                "phase_distribution_finish",
                "phase_distribution_direct",
                "phase_distribution_transition",
                "phase_distribution_quick_break",
                "phase_distribution_chaotic",
                "phase_distribution_set_play",
                "avg_phase_duration",
                "phases_leading_to_shot_pct",
                "phases_leading_to_goal_pct",
                "avg_team_width_in_possession",
                "avg_team_length_in_possession",
                "avg_team_width_out_of_possession",
                "avg_team_length_out_of_possession",
                "carries_in_create",
                "carries_in_build_up",
                "carries_in_finish",
                "carries_in_transition",
                "passes_into_space_in_create",
                "passes_into_space_in_finish",
                "pressures_in_transition",
                "progressive_actions_in_create",
                "progressive_actions_in_finish",
            ]
        )

    # Filter to midfielders if provided
    if midfielder_ids is not None:
        events_df = events_df[events_df["player_id"].isin(midfielder_ids)].copy()

    # Get unique player-match combinations from events
    player_matches = events_df[["player_id", "match_id"]].drop_duplicates()

    results = []

    for _, row in player_matches.iterrows():
        player_id = row["player_id"]
        match_id = row["match_id"]

        # Filter to this player-match
        player_match_events = events_df[
            (events_df["player_id"] == player_id) & (events_df["match_id"] == match_id)
        ]

        match_phases = phases_df[phases_df["match_id"] == match_id].copy()

        if player_match_events.empty or match_phases.empty:
            continue

        # Initialize phase counters
        phase_durations = []
        phase_types_count = {
            "create": 0,
            "build_up": 0,
            "finish": 0,
            "direct": 0,
            "transition": 0,
            "quick_break": 0,
            "chaotic": 0,
            "set_play": 0,
        }

        phases_leading_to_shot = 0
        phases_leading_to_goal = 0
        total_phases = 0

        team_width_in_possession = []
        team_length_in_possession = []
        team_width_out_of_possession = []
        team_length_out_of_possession = []

        # Event counts by phase
        carries_by_phase = {
            "create": 0,
            "build_up": 0,
            "finish": 0,
            "transition": 0,
        }
        passes_into_space_by_phase = {"create": 0, "finish": 0}
        pressures_in_transition = 0
        progressive_actions_by_phase = {"create": 0, "finish": 0}

        # Process each phase
        for _, phase in match_phases.iterrows():
            phase_type = phase.get("team_in_possession_phase_type", "")
            if pd.isna(phase_type) or phase_type == "":
                continue

            phase_type_lower = str(phase_type).lower()

            # Count phase types
            if phase_type_lower in phase_types_count:
                phase_types_count[phase_type_lower] += 1

            # Phase duration
            duration = phase.get("duration", 0)
            if not pd.isna(duration) and duration > 0:
                phase_durations.append(duration)

            # Phase outcomes
            if phase.get("team_possession_lead_to_shot", False):
                phases_leading_to_shot += 1
            if phase.get("team_possession_lead_to_goal", False):
                phases_leading_to_goal += 1
            total_phases += 1

            # Team shape metrics
            width_start = phase.get("team_in_possession_width_start")
            width_end = phase.get("team_in_possession_width_end")
            length_start = phase.get("team_in_possession_length_start")
            length_end = phase.get("team_in_possession_length_end")

            if not pd.isna(width_start):
                team_width_in_possession.append(width_start)
            if not pd.isna(width_end):
                team_width_in_possession.append(width_end)
            if not pd.isna(length_start):
                team_length_in_possession.append(length_start)
            if not pd.isna(length_end):
                team_length_in_possession.append(length_end)

            width_out_start = phase.get("team_out_of_possession_width_start")
            width_out_end = phase.get("team_out_of_possession_width_end")
            length_out_start = phase.get("team_out_of_possession_length_start")
            length_out_end = phase.get("team_out_of_possession_length_end")

            if not pd.isna(width_out_start):
                team_width_out_of_possession.append(width_out_start)
            if not pd.isna(width_out_end):
                team_width_out_of_possession.append(width_out_end)
            if not pd.isna(length_out_start):
                team_length_out_of_possession.append(length_out_start)
            if not pd.isna(length_out_end):
                team_length_out_of_possession.append(length_out_end)

            # Get events in this phase
            frame_start = phase.get("frame_start")
            frame_end = phase.get("frame_end")
            team_in_possession_id = phase.get("team_in_possession_id")

            if pd.isna(frame_start) or pd.isna(frame_end):
                continue

            # Find events that overlap with this phase
            # Event overlaps if: event_start <= phase_end AND event_end >= phase_start
            phase_events = player_match_events[
                (
                    player_match_events.get(
                        "frame_start", pd.Series([0] * len(player_match_events))
                    )
                    <= frame_end
                )
                & (
                    player_match_events.get(
                        "frame_end", pd.Series([0] * len(player_match_events))
                    )
                    >= frame_start
                )
            ]

            # Check if player is on the team in possession
            player_team_id = None
            if not phase_events.empty and "team_id" in phase_events.columns:
                player_team_id = (
                    phase_events["team_id"].iloc[0] if not phase_events.empty else None
                )

            is_in_possession_team = (
                player_team_id is not None
                and team_in_possession_id is not None
                and player_team_id == team_in_possession_id
            )

            # Count events by phase type (only for player's team in possession)
            if is_in_possession_team:
                # Carries
                if phase_type_lower in carries_by_phase:
                    carries = phase_events[
                        phase_events.get(
                            "carry", pd.Series([False] * len(phase_events))
                        )
                        == True
                    ]
                    carries_by_phase[phase_type_lower] += len(carries)

                # Passes into space
                if phase_type_lower in passes_into_space_by_phase:
                    passes = phase_events[
                        phase_events.get(
                            "end_type", pd.Series([""] * len(phase_events))
                        )
                        == "pass"
                    ]
                    passes_into_space = 0
                    if not passes.empty:
                        if "received_in_space" in passes.columns:
                            passes_into_space += passes["received_in_space"].sum()
                        if "pass_ahead" in passes.columns:
                            passes_into_space += passes["pass_ahead"].sum()
                    passes_into_space_by_phase[phase_type_lower] += passes_into_space

                # Progressive actions
                if phase_type_lower in progressive_actions_by_phase:
                    progressive = phase_events[
                        (
                            (
                                phase_events.get(
                                    "forward_momentum",
                                    pd.Series([False] * len(phase_events)),
                                )
                                == True
                            )
                            | (
                                phase_events.get(
                                    "break_defensive_line",
                                    pd.Series([False] * len(phase_events)),
                                )
                                == True
                            )
                        )
                    ]
                    progressive_actions_by_phase[phase_type_lower] += len(progressive)

            # Pressures (for transition phases, when opponent has possession)
            if phase_type_lower == "transition" and not is_in_possession_team:
                pressures = phase_events[
                    (
                        phase_events.get(
                            "event_type", pd.Series([""] * len(phase_events))
                        )
                        == "on_ball_engagement"
                    )
                    & (
                        phase_events.get(
                            "event_subtype", pd.Series([""] * len(phase_events))
                        ).isin(
                            ["pressing", "pressure", "recovery_press", "counter_press"]
                        )
                    )
                ]
                pressures_in_transition += len(pressures)

        # Calculate phase distribution (percentage of time in each phase)
        total_phase_count = sum(phase_types_count.values())
        phase_distribution = {
            f"phase_distribution_{k}": (
                (v / total_phase_count * 100) if total_phase_count > 0 else 0.0
            )
            for k, v in phase_types_count.items()
        }

        # Build result
        result = {
            "player_id": player_id,
            "match_id": match_id,
            **phase_distribution,
            "avg_phase_duration": np.mean(phase_durations) if phase_durations else 0.0,
            "phases_leading_to_shot_pct": (
                (phases_leading_to_shot / total_phases * 100)
                if total_phases > 0
                else 0.0
            ),
            "phases_leading_to_goal_pct": (
                (phases_leading_to_goal / total_phases * 100)
                if total_phases > 0
                else 0.0
            ),
            "avg_team_width_in_possession": (
                np.mean(team_width_in_possession) if team_width_in_possession else 0.0
            ),
            "avg_team_length_in_possession": (
                np.mean(team_length_in_possession) if team_length_in_possession else 0.0
            ),
            "avg_team_width_out_of_possession": (
                np.mean(team_width_out_of_possession)
                if team_width_out_of_possession
                else 0.0
            ),
            "avg_team_length_out_of_possession": (
                np.mean(team_length_out_of_possession)
                if team_length_out_of_possession
                else 0.0
            ),
            "carries_in_create": carries_by_phase.get("create", 0),
            "carries_in_build_up": carries_by_phase.get("build_up", 0),
            "carries_in_finish": carries_by_phase.get("finish", 0),
            "carries_in_transition": carries_by_phase.get("transition", 0),
            "passes_into_space_in_create": passes_into_space_by_phase.get("create", 0),
            "passes_into_space_in_finish": passes_into_space_by_phase.get("finish", 0),
            "pressures_in_transition": pressures_in_transition,
            "progressive_actions_in_create": progressive_actions_by_phase.get(
                "create", 0
            ),
            "progressive_actions_in_finish": progressive_actions_by_phase.get(
                "finish", 0
            ),
        }

        results.append(result)

    if not results:
        return pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "phase_distribution_create",
                "phase_distribution_build_up",
                "phase_distribution_finish",
                "phase_distribution_direct",
                "phase_distribution_transition",
                "phase_distribution_quick_break",
                "phase_distribution_chaotic",
                "phase_distribution_set_play",
                "avg_phase_duration",
                "phases_leading_to_shot_pct",
                "phases_leading_to_goal_pct",
                "avg_team_width_in_possession",
                "avg_team_length_in_possession",
                "avg_team_width_out_of_possession",
                "avg_team_length_out_of_possession",
                "carries_in_create",
                "carries_in_build_up",
                "carries_in_finish",
                "carries_in_transition",
                "passes_into_space_in_create",
                "passes_into_space_in_finish",
                "pressures_in_transition",
                "progressive_actions_in_create",
                "progressive_actions_in_finish",
            ]
        )

    features_df = pd.DataFrame(results)

    # Ensure numeric columns are properly typed
    numeric_cols = [
        col for col in features_df.columns if col not in ["player_id", "match_id"]
    ]
    for col in numeric_cols:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(
                0
            )

    return features_df


def load_and_extract_phase_features(
    phases_file: Path,
    events_df: pd.DataFrame,
    tracking_df: Optional[pd.DataFrame] = None,
    midfielder_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Load phases of play CSV and extract features.

    Args:
        phases_file: Path to phases of play CSV file
        events_df: DataFrame with dynamic events
        tracking_df: Optional DataFrame with tracking data
        midfielder_ids: Optional list of midfielder player IDs to filter

    Returns:
        DataFrame with phase-contextualized features
    """
    if not phases_file.exists():
        raise FileNotFoundError(f"Phases file not found: {phases_file}")

    phases_df = pd.read_csv(phases_file)

    return extract_phase_context_features(
        phases_df, events_df, tracking_df, midfielder_ids
    )


def extract_phase_features_from_multiple_matches(
    phases_files: List[Path],
    events_dfs: List[pd.DataFrame],
    tracking_dfs: Optional[List[pd.DataFrame]] = None,
    midfielder_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extract phase features from multiple match files.

    Args:
        phases_files: List of paths to phases of play CSV files
        events_dfs: List of DataFrames with dynamic events (one per match)
        tracking_dfs: Optional list of DataFrames with tracking data
        midfielder_ids: Optional list of midfielder player IDs to filter

    Returns:
        Combined DataFrame with phase features from all matches
    """
    all_features = []

    for i, phases_file in enumerate(phases_files):
        if not phases_file.exists():
            continue

        if i >= len(events_dfs):
            continue

        events_df = events_dfs[i]
        tracking_df = (
            tracking_dfs[i] if tracking_dfs and i < len(tracking_dfs) else None
        )

        try:
            features = load_and_extract_phase_features(
                phases_file, events_df, tracking_df, midfielder_ids
            )
            if not features.empty:
                all_features.append(features)
        except Exception as e:
            print(f"Warning: Could not process {phases_file}: {e}")
            continue

    if not all_features:
        return pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "phase_distribution_create",
                "phase_distribution_build_up",
                "phase_distribution_finish",
                "phase_distribution_direct",
                "phase_distribution_transition",
                "phase_distribution_quick_break",
                "phase_distribution_chaotic",
                "phase_distribution_set_play",
                "avg_phase_duration",
                "phases_leading_to_shot_pct",
                "phases_leading_to_goal_pct",
                "avg_team_width_in_possession",
                "avg_team_length_in_possession",
                "avg_team_width_out_of_possession",
                "avg_team_length_out_of_possession",
                "carries_in_create",
                "carries_in_build_up",
                "carries_in_finish",
                "carries_in_transition",
                "passes_into_space_in_create",
                "passes_into_space_in_finish",
                "pressures_in_transition",
                "progressive_actions_in_create",
                "progressive_actions_in_finish",
            ]
        )

    return pd.concat(all_features, ignore_index=True)
