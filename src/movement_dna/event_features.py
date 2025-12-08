"""
Event Feature Extraction for Movement DNA

Extracts per-match event features for midfielders using event_type/event_type_id and event_subtype/event_subtype_id.
Supports separate extraction of OBR (Off-Ball Run), PO (Passing Option), PP (Player Possession), and OBE (On-Ball Engagement) events.

Event Type IDs:
- 1 = off_ball_run (OBR)
- 7 = passing_option (PO)
- 8 = player_possession (PP)
- 9 = on_ball_engagement (OBE) (event_type string: defensive_engagement)

Event Subtype IDs:
- 1 = behind, 2 = coming_short, 3 = cross_receiver, 5 = overlap,
- 8 = run_ahead_of_the_ball, 9 = support, 10 = underlap,
- 11 = pressing, 12 = pressure, 13 = counter_press, 14 = recovery_press, 15 = other
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Literal


def _match_event_type(df: pd.DataFrame, event_type: Union[str, int]) -> pd.Series:
    """
    Match events by event_type (string) or event_type_id (int).

    Args:
        df: DataFrame with events
        event_type: Event type as string or int ID

    Returns:
        Boolean Series indicating matches
    """
    if isinstance(event_type, int):
        # Match by event_type_id
        if "event_type_id" in df.columns:
            return df["event_type_id"] == event_type
        else:
            # Fallback: map ID to string
            id_to_type = {
                1: "off_ball_run",
                7: "passing_option",
                8: "player_possession",
                9: "defensive_engagement",  # or "on_ball_engagement"
            }
            event_type_str = id_to_type.get(event_type, "")
            if "event_type" in df.columns:
                return df["event_type"] == event_type_str
    else:
        # Match by event_type string
        if "event_type" in df.columns:
            mask = df["event_type"] == event_type
            # Also check event_type_id if available
            if "event_type_id" in df.columns:
                type_to_id = {
                    "off_ball_run": 1,
                    "passing_option": 7,
                    "player_possession": 8,
                    "defensive_engagement": 9,
                    "on_ball_engagement": 9,  # Support both names
                }
                event_type_id = type_to_id.get(event_type)
                if event_type_id is not None:
                    mask = mask | (df["event_type_id"] == event_type_id)
            return mask

    return pd.Series([False] * len(df))


def _match_event_subtype(df: pd.DataFrame, subtype: Union[str, int]) -> pd.Series:
    """
    Match events by event_subtype (string) or event_subtype_id (int).

    Args:
        df: DataFrame with events
        subtype: Event subtype as string or int ID

    Returns:
        Boolean Series indicating matches
    """
    if isinstance(subtype, int):
        # Match by event_subtype_id
        if "event_subtype_id" in df.columns:
            return df["event_subtype_id"] == subtype
        else:
            # Fallback: map ID to string
            id_to_subtype = {
                1: "behind",
                2: "coming_short",
                3: "cross_receiver",
                5: "overlap",
                8: "run_ahead_of_the_ball",
                9: "support",
                10: "underlap",
                11: "pressing",
                12: "pressure",
                13: "counter_press",
                14: "recovery_press",
                15: "other",
            }
            subtype_str = id_to_subtype.get(subtype, "")
            if "event_subtype" in df.columns:
                return df["event_subtype"] == subtype_str
    else:
        # Match by event_subtype string
        if "event_subtype" in df.columns:
            mask = df["event_subtype"] == subtype
            # Also check event_subtype_id if available
            if "event_subtype_id" in df.columns:
                subtype_to_id = {
                    "behind": 1,
                    "coming_short": 2,
                    "cross_receiver": 3,
                    "overlap": 5,
                    "run_ahead_of_the_ball": 8,
                    "support": 9,
                    "underlap": 10,
                    "pressing": 11,
                    "pressure": 12,
                    "counter_press": 13,
                    "recovery_press": 14,
                    "other": 15,
                }
                subtype_id = subtype_to_id.get(subtype)
                if subtype_id is not None:
                    mask = mask | (df["event_subtype_id"] == subtype_id)
            return mask

    return pd.Series([False] * len(df))


def _match_event_subtypes(
    df: pd.DataFrame, subtypes: List[Union[str, int]]
) -> pd.Series:
    """
    Match events by multiple event_subtype values (string or int IDs).

    Args:
        df: DataFrame with events
        subtypes: List of event subtypes as strings or int IDs

    Returns:
        Boolean Series indicating matches
    """
    mask = pd.Series([False] * len(df))
    for subtype in subtypes:
        mask = mask | _match_event_subtype(df, subtype)
    return mask


def extract_obr_events(
    events_df: pd.DataFrame,
    player_ids: Optional[List[int]] = None,
    include_attributes: bool = True,
) -> pd.DataFrame:
    """
    Extract Off-Ball Run (OBR) events.

    Args:
        events_df: DataFrame with dynamic events (must include match_id column)
        player_ids: Optional list of player IDs to filter. If None, includes all players.
        include_attributes: If True, includes additional attributes from CSV specification

    Returns:
        DataFrame with OBR events and their attributes
    """
    if events_df.empty:
        return pd.DataFrame()

    # Ensure match_id exists
    if "match_id" not in events_df.columns:
        raise ValueError("events_df must contain 'match_id' column")

    # Filter to OBR events
    obr_mask = _match_event_type(events_df, "off_ball_run") | _match_event_type(
        events_df, 1
    )
    obr_events = events_df[obr_mask].copy()

    # Filter by player_ids if provided
    if player_ids is not None:
        obr_events = obr_events[obr_events["player_id"].isin(player_ids)].copy()

    if obr_events.empty:
        return pd.DataFrame()

    # Select relevant columns based on CSV specification
    base_columns = [
        "event_id",
        "match_id",
        "player_id",
        "player_name",
        "event_type",
        "event_type_id",
        "event_subtype",
        "event_subtype_id",
    ]

    if include_attributes:
        # Add OBR-specific attributes
        obr_specific = [
            "channel_id_end",
            "channel_end",
            "associated_player_possession_event_id",
            "associated_player_possession_frame_start",
            "player_in_possession_id",
            "player_in_possession_name",
            "n_simultaneous_runs",
            "give_and_go",
            "intended_run_behind",
            "push_defensive_line",
            "break_defensive_line",
            "passing_option_at_start",
            "n_opponents_ahead_end",
            "n_opponents_ahead_start",
            "n_opponents_overtaken",
        ]
        base_columns.extend(obr_specific)

    # Select only columns that exist in the DataFrame
    available_columns = [col for col in base_columns if col in obr_events.columns]
    return obr_events[available_columns].copy()


def extract_po_events(
    events_df: pd.DataFrame,
    player_ids: Optional[List[int]] = None,
    include_attributes: bool = True,
) -> pd.DataFrame:
    """
    Extract Passing Option (PO) events.

    Args:
        events_df: DataFrame with dynamic events (must include match_id column)
        player_ids: Optional list of player IDs to filter. If None, includes all players.
        include_attributes: If True, includes additional attributes from CSV specification

    Returns:
        DataFrame with PO events and their attributes
    """
    if events_df.empty:
        return pd.DataFrame()

    # Ensure match_id exists
    if "match_id" not in events_df.columns:
        raise ValueError("events_df must contain 'match_id' column")

    # Filter to PO events
    po_mask = _match_event_type(events_df, "passing_option") | _match_event_type(
        events_df, 7
    )
    po_events = events_df[po_mask].copy()

    # Filter by player_ids if provided
    if player_ids is not None:
        po_events = po_events[po_events["player_id"].isin(player_ids)].copy()

    if po_events.empty:
        return pd.DataFrame()

    # Select relevant columns based on CSV specification
    base_columns = [
        "event_id",
        "match_id",
        "player_id",
        "player_name",
        "event_type",
        "event_type_id",
    ]

    if include_attributes:
        # Add PO-specific attributes
        po_specific = [
            "channel_id_end",
            "channel_end",
            "associated_player_possession_event_id",
            "associated_player_possession_frame_start",
            "associated_off_ball_run_event_id",
            "associated_off_ball_run_subtype_id",
            "associated_off_ball_run_subtype",
            "player_in_possession_id",
            "player_in_possession_name",
            "targeted",
            "received",
            "received_in_space",
            "passing_option_at_player_possession_start",
            "peak_passing_option_frame",
            "n_simultaneous_passing_options",
            "first_line_break_type_id",
            "first_line_break_type",
            "second_last_line_break_type_id",
            "second_last_line_break_type",
            "last_line_break_type_id",
            "last_line_break_type",
            "high_pass",
            "n_opponents_ahead_end",
            "n_opponents_ahead_start",
            "n_opponents_overtaken",
        ]
        base_columns.extend(po_specific)

    # Select only columns that exist in the DataFrame
    available_columns = [col for col in base_columns if col in po_events.columns]
    return po_events[available_columns].copy()


def extract_pp_events(
    events_df: pd.DataFrame,
    player_ids: Optional[List[int]] = None,
    include_attributes: bool = True,
) -> pd.DataFrame:
    """
    Extract Player Possession (PP) events.

    Args:
        events_df: DataFrame with dynamic events (must include match_id column)
        player_ids: Optional list of player IDs to filter. If None, includes all players.
        include_attributes: If True, includes additional attributes from CSV specification

    Returns:
        DataFrame with PP events and their attributes
    """
    if events_df.empty:
        return pd.DataFrame()

    # Ensure match_id exists
    if "match_id" not in events_df.columns:
        raise ValueError("events_df must contain 'match_id' column")

    # Filter to PP events
    pp_mask = _match_event_type(events_df, "player_possession") | _match_event_type(
        events_df, 8
    )
    pp_events = events_df[pp_mask].copy()

    # Filter by player_ids if provided
    if player_ids is not None:
        pp_events = pp_events[pp_events["player_id"].isin(player_ids)].copy()

    if pp_events.empty:
        return pd.DataFrame()

    # Select relevant columns based on CSV specification
    base_columns = [
        "event_id",
        "match_id",
        "player_id",
        "player_name",
        "event_type",
        "event_type_id",
    ]

    if include_attributes:
        # Add PP-specific attributes
        pp_specific = [
            "channel_id_end",
            "channel_end",
            "associated_player_possession_frame_end",
            "associated_player_possession_end_type_id",
            "associated_player_possession_end_type",
            "game_interruption_before_id",
            "game_interruption_before",
            "game_interruption_after_id",
            "game_interruption_after",
            "start_type_id",
            "start_type",
            "end_type_id",
            "end_type",
            "carry",
            "forward_momentum",
            "break_defensive_line",
            "pass_ahead",
            "received_in_space",
        ]
        base_columns.extend(pp_specific)

    # Select only columns that exist in the DataFrame
    available_columns = [col for col in base_columns if col in pp_events.columns]
    return pp_events[available_columns].copy()


def extract_obe_events(
    events_df: pd.DataFrame,
    player_ids: Optional[List[int]] = None,
    include_attributes: bool = True,
) -> pd.DataFrame:
    """
    Extract On-Ball Engagement (OBE) events.

    Args:
        events_df: DataFrame with dynamic events (must include match_id column)
        player_ids: Optional list of player IDs to filter. If None, includes all players.
        include_attributes: If True, includes additional attributes from CSV specification

    Returns:
        DataFrame with OBE events and their attributes
    """
    if events_df.empty:
        return pd.DataFrame()

    # Ensure match_id exists
    if "match_id" not in events_df.columns:
        raise ValueError("events_df must contain 'match_id' column")

    # Filter to OBE events (support both names)
    obe_mask = (
        _match_event_type(events_df, "defensive_engagement")
        | _match_event_type(events_df, "on_ball_engagement")
        | _match_event_type(events_df, 9)
    )
    obe_events = events_df[obe_mask].copy()

    # Filter by player_ids if provided
    if player_ids is not None:
        obe_events = obe_events[obe_events["player_id"].isin(player_ids)].copy()

    if obe_events.empty:
        return pd.DataFrame()

    # Select relevant columns based on CSV specification
    base_columns = [
        "event_id",
        "match_id",
        "player_id",
        "player_name",
        "event_type",
        "event_type_id",
        "event_subtype",
        "event_subtype_id",
    ]

    if include_attributes:
        # Add OBE-specific attributes
        obe_specific = [
            "frame_physical_start",
            "possession_danger",
            "beaten_by_possession",
            "beaten_by_movement",
            "stop_possession_danger",
            "reduce_possession_danger",
            "force_backward",
            "player_targeted_id",
            "player_targeted_name",
            "affected_line_breaking_passing_option_id",
            "affected_line_break_id",
            "affected_line_break",
            "affected_line_breaking_passing_option_attempted",
            "affected_line_breaking_passing_option_xthreat",
            "affected_line_breaking_passing_option_dangerous",
            "affected_line_breaking_passing_option_run_subtype_id",
            "affected_line_breaking_passing_option_run_subtype",
            "pressing_chain",
            "pressing_chain_length",
            "pressing_chain_end_type_id",
            "pressing_chain_end_type",
            "pressing_chain_index",
            "index_in_pressing_chain",
        ]
        base_columns.extend(obe_specific)

    # Select only columns that exist in the DataFrame
    available_columns = [col for col in base_columns if col in obe_events.columns]
    return obe_events[available_columns].copy()


def extract_event_features(
    events_df: pd.DataFrame,
    midfielder_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extract aggregated event-based features per player per match using event_type and event_subtype.

    Args:
        events_df: DataFrame with dynamic events (must include match_id column)
        midfielder_ids: Optional list of midfielder player IDs to filter. If None, includes all players.

    Returns:
        DataFrame with columns:
        - Basic: player_id, match_id
        - Carries: carries_count, carries_distance
        - Passes: passes_into_space
        - Pressures: pressures (total), pressing_count, pressure_count, recovery_press_count, counter_press_count
        - Progressive: progressive_actions
        - Off-ball runs: off_ball_runs_count, off_ball_runs_behind, off_ball_runs_coming_short,
          off_ball_runs_cross_receiver, off_ball_runs_overlap, off_ball_runs_run_ahead,
          off_ball_runs_support, off_ball_runs_underlap
        - Passing options: passing_options_count, passing_options_targeted, passing_options_received
    """
    if events_df.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "carries_count",
                "carries_distance",
                "passes_into_space",
                "pressures",
                "progressive_actions",
                "pressing_count",
                "pressure_count",
                "recovery_press_count",
                "counter_press_count",
                "off_ball_runs_count",
                "off_ball_runs_behind",
                "off_ball_runs_coming_short",
                "off_ball_runs_cross_receiver",
                "off_ball_runs_overlap",
                "off_ball_runs_run_ahead",
                "off_ball_runs_support",
                "off_ball_runs_underlap",
                "passing_options_count",
                "passing_options_targeted",
                "passing_options_received",
            ]
        )

    # Ensure match_id exists
    if "match_id" not in events_df.columns:
        raise ValueError("events_df must contain 'match_id' column")

    # Filter to midfielders if provided
    if midfielder_ids is not None:
        # Filter events where player_id is in midfielder_ids
        player_possession_mask = _match_event_type(
            events_df, "player_possession"
        ) | _match_event_type(events_df, 8)
        player_possession_events = events_df[
            player_possession_mask & (events_df["player_id"].isin(midfielder_ids))
        ].copy()

        # For other event types, filter by player_id
        other_events = events_df[
            ~player_possession_mask & (events_df["player_id"].isin(midfielder_ids))
        ].copy()
    else:
        player_possession_mask = _match_event_type(
            events_df, "player_possession"
        ) | _match_event_type(events_df, 8)
        player_possession_events = events_df[player_possession_mask].copy()
        other_events = events_df[~player_possession_mask].copy()

    # Initialize result list
    results = []

    # Get unique player-match combinations
    if not player_possession_events.empty:
        player_matches = player_possession_events[
            ["player_id", "match_id"]
        ].drop_duplicates()
    elif not other_events.empty:
        player_matches = other_events[["player_id", "match_id"]].drop_duplicates()
    else:
        return pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "carries_count",
                "carries_distance",
                "passes_into_space",
                "pressures",
                "progressive_actions",
                "pressing_count",
                "pressure_count",
                "recovery_press_count",
                "counter_press_count",
                "off_ball_runs_count",
                "off_ball_runs_behind",
                "off_ball_runs_coming_short",
                "off_ball_runs_cross_receiver",
                "off_ball_runs_overlap",
                "off_ball_runs_run_ahead",
                "off_ball_runs_support",
                "off_ball_runs_underlap",
                "passing_options_count",
                "passing_options_targeted",
                "passing_options_received",
            ]
        )

    # Process each player-match combination
    for _, row in player_matches.iterrows():
        player_id = row["player_id"]
        match_id = row["match_id"]

        # Filter events for this player-match
        player_match_possession = player_possession_events[
            (player_possession_events["player_id"] == player_id)
            & (player_possession_events["match_id"] == match_id)
        ]

        player_match_other = other_events[
            (other_events["player_id"] == player_id)
            & (other_events["match_id"] == match_id)
        ]

        # 1. CARRIES: Count and distance
        carries = player_match_possession[
            player_match_possession.get(
                "carry", pd.Series([False] * len(player_match_possession))
            )
            == True
        ]
        carries_count = len(carries)
        # Distance covered during carries
        if "distance_covered" in carries.columns:
            carries_distance = carries["distance_covered"].sum()
        else:
            # Fallback: estimate from start/end positions if available
            if "x_start" in carries.columns and "x_end" in carries.columns:
                carries["carry_distance"] = np.sqrt(
                    (carries["x_end"] - carries["x_start"]) ** 2
                    + (carries["y_end"] - carries["y_start"]) ** 2
                )
                carries_distance = carries["carry_distance"].sum()
            else:
                carries_distance = 0.0

        # 2. PASSES INTO SPACE
        # Passes where received_in_space == True or pass_ahead == True
        passes = player_match_possession[player_match_possession["end_type"] == "pass"]

        passes_into_space = 0
        if not passes.empty:
            # Check for received_in_space (for passes received)
            if "received_in_space" in passes.columns:
                passes_into_space += passes["received_in_space"].sum()

            # Check for pass_ahead (for passes made)
            if "pass_ahead" in passes.columns:
                passes_into_space += passes["pass_ahead"].sum()

            # Also check passing_option events where received_in_space might be True
            passing_option_mask = _match_event_type(
                events_df, "passing_option"
            ) | _match_event_type(events_df, 7)
            passing_options = events_df[
                passing_option_mask
                & (events_df["player_in_possession_id"] == player_id)
                & (events_df["match_id"] == match_id)
            ]
            if (
                not passing_options.empty
                and "received_in_space" in passing_options.columns
            ):
                passes_into_space += passing_options["received_in_space"].sum()

        # 3. PRESSURES (with subtypes)
        # Get all defensive_engagement/on_ball_engagement events for this player
        # Support both event_type names and event_type_id = 9
        defensive_engagement_mask = (
            _match_event_type(player_match_other, "defensive_engagement")
            | _match_event_type(player_match_other, "on_ball_engagement")
            | _match_event_type(player_match_other, 9)
        )
        on_ball_engagements = player_match_other[defensive_engagement_mask]

        # Total pressures (all pressure subtypes)
        # Subtypes: pressing (11), pressure (12), recovery_press (14), counter_press (13)
        pressure_subtypes = ["pressing", "pressure", "recovery_press", "counter_press"]
        pressure_subtype_ids = [11, 12, 14, 13]
        pressures = on_ball_engagements[
            _match_event_subtypes(
                on_ball_engagements, pressure_subtypes + pressure_subtype_ids
            )
        ]
        pressures_count = len(pressures)

        # Pressure subtypes (granular)
        pressing_count = len(
            on_ball_engagements[
                _match_event_subtype(on_ball_engagements, "pressing")
                | _match_event_subtype(on_ball_engagements, 11)
            ]
        )
        pressure_count = len(
            on_ball_engagements[
                _match_event_subtype(on_ball_engagements, "pressure")
                | _match_event_subtype(on_ball_engagements, 12)
            ]
        )
        recovery_press_count = len(
            on_ball_engagements[
                _match_event_subtype(on_ball_engagements, "recovery_press")
                | _match_event_subtype(on_ball_engagements, 14)
            ]
        )
        counter_press_count = len(
            on_ball_engagements[
                _match_event_subtype(on_ball_engagements, "counter_press")
                | _match_event_subtype(on_ball_engagements, 13)
            ]
        )

        # 4. PROGRESSIVE ACTIONS
        # Carries/passes with forward_momentum == True or break_defensive_line == True
        # Count unique events that have at least one progressive indicator
        progressive_actions = 0
        if not player_match_possession.empty:
            progressive_mask = (
                player_match_possession.get(
                    "forward_momentum",
                    pd.Series([False] * len(player_match_possession)),
                )
                == True
            ) | (
                player_match_possession.get(
                    "break_defensive_line",
                    pd.Series([False] * len(player_match_possession)),
                )
                == True
            )
            progressive_actions = progressive_mask.sum()

        # 5. OFF-BALL RUNS (by subtype)
        # Get off_ball_run events for this player (event_type_id = 1)
        off_ball_run_mask = _match_event_type(
            player_match_other, "off_ball_run"
        ) | _match_event_type(player_match_other, 1)
        off_ball_runs = player_match_other[off_ball_run_mask]

        off_ball_runs_count = len(off_ball_runs)
        # Off-ball run subtypes with both string and ID matching
        # Note: dropping_off, pulling_half_space, pulling_wide removed per user request
        off_ball_runs_behind = len(
            off_ball_runs[
                _match_event_subtype(off_ball_runs, "behind")
                | _match_event_subtype(off_ball_runs, 1)
            ]
        )
        off_ball_runs_coming_short = len(
            off_ball_runs[
                _match_event_subtype(off_ball_runs, "coming_short")
                | _match_event_subtype(off_ball_runs, 2)
            ]
        )
        off_ball_runs_cross_receiver = len(
            off_ball_runs[
                _match_event_subtype(off_ball_runs, "cross_receiver")
                | _match_event_subtype(off_ball_runs, 3)
            ]
        )
        off_ball_runs_overlap = len(
            off_ball_runs[
                _match_event_subtype(off_ball_runs, "overlap")
                | _match_event_subtype(off_ball_runs, 5)
            ]
        )
        off_ball_runs_run_ahead = len(
            off_ball_runs[
                _match_event_subtype(off_ball_runs, "run_ahead_of_the_ball")
                | _match_event_subtype(off_ball_runs, 8)
            ]
        )
        off_ball_runs_support = len(
            off_ball_runs[
                _match_event_subtype(off_ball_runs, "support")
                | _match_event_subtype(off_ball_runs, 9)
            ]
        )
        off_ball_runs_underlap = len(
            off_ball_runs[
                _match_event_subtype(off_ball_runs, "underlap")
                | _match_event_subtype(off_ball_runs, 10)
            ]
        )

        # 6. PASSING OPTIONS
        # Count passing options where player was a target (event_type_id = 7)
        passing_option_mask = _match_event_type(
            events_df, "passing_option"
        ) | _match_event_type(events_df, 7)
        passing_options = events_df[
            passing_option_mask
            & (events_df["player_id"] == player_id)
            & (events_df["match_id"] == match_id)
        ]
        passing_options_count = len(passing_options)
        passing_options_targeted = len(
            passing_options[
                passing_options.get(
                    "targeted", pd.Series([False] * len(passing_options))
                )
                == True
            ]
        )
        passing_options_received = len(
            passing_options[
                passing_options.get(
                    "received", pd.Series([False] * len(passing_options))
                )
                == True
            ]
        )

        # Store results
        results.append(
            {
                "player_id": player_id,
                "match_id": match_id,
                # Original features
                "carries_count": carries_count,
                "carries_distance": carries_distance,
                "passes_into_space": int(passes_into_space),
                "pressures": pressures_count,
                "progressive_actions": int(progressive_actions),
                # Pressure subtypes
                "pressing_count": pressing_count,
                "pressure_count": pressure_count,
                "recovery_press_count": recovery_press_count,
                "counter_press_count": counter_press_count,
                # Off-ball runs
                "off_ball_runs_count": off_ball_runs_count,
                "off_ball_runs_behind": off_ball_runs_behind,
                "off_ball_runs_coming_short": off_ball_runs_coming_short,
                "off_ball_runs_cross_receiver": off_ball_runs_cross_receiver,
                "off_ball_runs_overlap": off_ball_runs_overlap,
                "off_ball_runs_run_ahead": off_ball_runs_run_ahead,
                "off_ball_runs_support": off_ball_runs_support,
                "off_ball_runs_underlap": off_ball_runs_underlap,
                # Passing options
                "passing_options_count": passing_options_count,
                "passing_options_targeted": passing_options_targeted,
                "passing_options_received": passing_options_received,
            }
        )

    # Create DataFrame
    if not results:
        return pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "carries_count",
                "carries_distance",
                "passes_into_space",
                "pressures",
                "progressive_actions",
                "pressing_count",
                "pressure_count",
                "recovery_press_count",
                "counter_press_count",
                "off_ball_runs_count",
                "off_ball_runs_behind",
                "off_ball_runs_coming_short",
                "off_ball_runs_cross_receiver",
                "off_ball_runs_overlap",
                "off_ball_runs_run_ahead",
                "off_ball_runs_support",
                "off_ball_runs_underlap",
                "passing_options_count",
                "passing_options_targeted",
                "passing_options_received",
            ]
        )

    features_df = pd.DataFrame(results)

    # Ensure all numeric columns are properly typed
    numeric_cols = [
        "carries_count",
        "carries_distance",
        "passes_into_space",
        "pressures",
        "progressive_actions",
        "pressing_count",
        "pressure_count",
        "recovery_press_count",
        "counter_press_count",
        "off_ball_runs_count",
        "off_ball_runs_behind",
        "off_ball_runs_coming_short",
        "off_ball_runs_cross_receiver",
        "off_ball_runs_overlap",
        "off_ball_runs_run_ahead",
        "off_ball_runs_support",
        "off_ball_runs_underlap",
        "passing_options_count",
        "passing_options_targeted",
        "passing_options_received",
    ]
    for col in numeric_cols:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(
                0
            )

    return features_df


def extract_event_features_from_file(
    events_file: Path,
    midfielder_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Load events from file and extract features.

    Args:
        events_file: Path to dynamic events CSV file
        midfielder_ids: Optional list of midfielder player IDs to filter

    Returns:
        DataFrame with event features
    """
    events_df = pd.read_csv(events_file)

    # Extract match_id from filename if not in DataFrame
    if "match_id" not in events_df.columns:
        # Try to extract from filename (e.g., "1886347_dynamic_events.csv")
        match_id = int(events_file.stem.split("_")[0])
        events_df["match_id"] = match_id

    return extract_event_features(events_df, midfielder_ids)


def extract_event_features_from_multiple_matches(
    events_files: List[Path],
    midfielder_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extract event features from multiple match files.

    Args:
        events_files: List of paths to dynamic events CSV files
        midfielder_ids: Optional list of midfielder player IDs to filter

    Returns:
        Combined DataFrame with event features from all matches
    """
    all_features = []

    for events_file in events_files:
        if not events_file.exists():
            continue

        try:
            features = extract_event_features_from_file(events_file, midfielder_ids)
            if not features.empty:
                all_features.append(features)
        except Exception as e:
            print(f"Warning: Could not process {events_file}: {e}")
            continue

    if not all_features:
        return pd.DataFrame(
            columns=[
                "player_id",
                "match_id",
                "carries_count",
                "carries_distance",
                "passes_into_space",
                "pressures",
                "progressive_actions",
                "pressing_count",
                "pressure_count",
                "recovery_press_count",
                "counter_press_count",
                "off_ball_runs_count",
                "off_ball_runs_behind",
                "off_ball_runs_coming_short",
                "off_ball_runs_cross_receiver",
                "off_ball_runs_overlap",
                "off_ball_runs_run_ahead",
                "off_ball_runs_support",
                "off_ball_runs_underlap",
                "passing_options_count",
                "passing_options_targeted",
                "passing_options_received",
            ]
        )

    return pd.concat(all_features, ignore_index=True)


def extract_events_by_type(
    events_df: pd.DataFrame,
    event_type: Literal["OBR", "PO", "PP", "OBE"],
    player_ids: Optional[List[int]] = None,
    include_attributes: bool = True,
) -> pd.DataFrame:
    """
    Extract events by type (OBR, PO, PP, or OBE).

    Args:
        events_df: DataFrame with dynamic events (must include match_id column)
        event_type: Type of events to extract: "OBR", "PO", "PP", or "OBE"
        player_ids: Optional list of player IDs to filter. If None, includes all players.
        include_attributes: If True, includes additional attributes from CSV specification

    Returns:
        DataFrame with events of the specified type
    """
    event_type_map = {
        "OBR": extract_obr_events,
        "PO": extract_po_events,
        "PP": extract_pp_events,
        "OBE": extract_obe_events,
    }

    if event_type not in event_type_map:
        raise ValueError(
            f"event_type must be one of {list(event_type_map.keys())}, got {event_type}"
        )

    return event_type_map[event_type](events_df, player_ids, include_attributes)
