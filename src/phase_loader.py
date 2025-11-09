"""
Phase of play data loading and linking utilities.

This module provides functions to load phases of play data and link
pressing events to phase context.
"""

from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
from .data_loader import load_phases_of_play


def link_pressing_to_phase(
    pressing_events: pd.DataFrame, phases_of_play: pd.DataFrame
) -> pd.DataFrame:
    """
    Link pressing events to their corresponding phases of play.

    Parameters
    ----------
    pressing_events : pd.DataFrame
        Pressing events DataFrame (must have 'frame_start' column)
    phases_of_play : pd.DataFrame
        Phases of play DataFrame (must have 'frame_start' and 'frame_end' columns)

    Returns
    -------
    pd.DataFrame
        Pressing events with phase context added
    """
    pressing_with_phase = pressing_events.copy()

    # Create a mapping from frame to phase
    phase_map = []
    for _, phase in phases_of_play.iterrows():
        phase_start = phase["frame_start"]
        phase_end = phase["frame_end"]

        # Add phase information for all frames in this phase
        phase_map.append(
            {
                "frame_start": phase_start,
                "frame_end": phase_end,
                "phase_index": phase.get("index", phase.name),
                "team_in_possession_phase_type": phase.get(
                    "team_in_possession_phase_type"
                ),
                "team_out_of_possession_phase_type": phase.get(
                    "team_out_of_possession_phase_type"
                ),
                "team_in_possession_id": phase.get("team_in_possession_id"),
                "team_in_possession_shortname": phase.get(
                    "team_in_possession_shortname"
                ),
                "attacking_side": phase.get("attacking_side"),
                "team_possession_lead_to_goal": phase.get(
                    "team_possession_lead_to_goal", False
                ),
                "team_possession_lead_to_shot": phase.get(
                    "team_possession_lead_to_shot", False
                ),
                "x_start": phase.get("x_start"),
                "y_start": phase.get("y_start"),
                "x_end": phase.get("x_end"),
                "y_end": phase.get("y_end"),
                "third_start": phase.get("third_start"),
                "third_end": phase.get("third_end"),
                "channel_start": phase.get("channel_start"),
                "channel_end": phase.get("channel_end"),
            }
        )

    phase_df = pd.DataFrame(phase_map)

    # Merge pressing events with phases based on frame overlap
    pressing_with_phase_list = []

    for _, pressing_event in pressing_with_phase.iterrows():
        event_frame_start = pressing_event.get("frame_start")
        event_frame_end = pressing_event.get("frame_end", event_frame_start)

        # Find phases that overlap with this event
        overlapping_phases = phase_df[
            (phase_df["frame_start"] <= event_frame_end)
            & (phase_df["frame_end"] >= event_frame_start)
        ]

        if len(overlapping_phases) > 0:
            # Use the first overlapping phase (or the one with most overlap)
            phase_info = overlapping_phases.iloc[0]

            # Create merged row
            merged_row = pressing_event.to_dict()
            merged_row.update(
                {
                    "phase_index": phase_info["phase_index"],
                    "phase_team_in_possession_phase_type": phase_info[
                        "team_in_possession_phase_type"
                    ],
                    "phase_team_out_of_possession_phase_type": phase_info[
                        "team_out_of_possession_phase_type"
                    ],
                    "phase_team_in_possession_id": phase_info["team_in_possession_id"],
                    "phase_team_in_possession_shortname": phase_info[
                        "team_in_possession_shortname"
                    ],
                    "phase_attacking_side": phase_info["attacking_side"],
                    "phase_lead_to_goal": phase_info["team_possession_lead_to_goal"],
                    "phase_lead_to_shot": phase_info["team_possession_lead_to_shot"],
                    "phase_x_start": phase_info["x_start"],
                    "phase_y_start": phase_info["y_start"],
                    "phase_x_end": phase_info["x_end"],
                    "phase_y_end": phase_info["y_end"],
                    "phase_third_start": phase_info["third_start"],
                    "phase_third_end": phase_info["third_end"],
                    "phase_channel_start": phase_info["channel_start"],
                    "phase_channel_end": phase_info["channel_end"],
                }
            )
        else:
            # No overlapping phase found, keep original event
            merged_row = pressing_event.to_dict()
            merged_row.update(
                {
                    "phase_index": None,
                    "phase_team_in_possession_phase_type": None,
                    "phase_team_out_of_possession_phase_type": None,
                    "phase_team_in_possession_id": None,
                    "phase_team_in_possession_shortname": None,
                    "phase_attacking_side": None,
                    "phase_lead_to_goal": False,
                    "phase_lead_to_shot": False,
                    "phase_x_start": None,
                    "phase_y_start": None,
                    "phase_x_end": None,
                    "phase_y_end": None,
                    "phase_third_start": None,
                    "phase_third_end": None,
                    "phase_channel_start": None,
                    "phase_channel_end": None,
                }
            )

        pressing_with_phase_list.append(merged_row)

    return pd.DataFrame(pressing_with_phase_list)


def get_phases_by_type(
    phases_of_play: pd.DataFrame,
    phase_type: Optional[str] = None,
    defensive_phase_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter phases of play by type.

    Parameters
    ----------
    phases_of_play : pd.DataFrame
        Phases of play DataFrame
    phase_type : str, optional
        Filter by team in possession phase type (e.g., 'build_up', 'create', 'finish')
    defensive_phase_type : str, optional
        Filter by team out of possession phase type (e.g., 'high_block', 'medium_block', 'low_block')

    Returns
    -------
    pd.DataFrame
        Filtered phases of play
    """
    filtered = phases_of_play.copy()

    if phase_type is not None:
        filtered = filtered[filtered["team_in_possession_phase_type"] == phase_type]

    if defensive_phase_type is not None:
        filtered = filtered[
            filtered["team_out_of_possession_phase_type"] == defensive_phase_type
        ]

    return filtered


def get_phases_by_frame_range(
    phases_of_play: pd.DataFrame, frame_start: int, frame_end: int
) -> pd.DataFrame:
    """
    Get phases of play that overlap with a frame range.

    Parameters
    ----------
    phases_of_play : pd.DataFrame
        Phases of play DataFrame
    frame_start : int
        Start frame
    frame_end : int
        End frame

    Returns
    -------
    pd.DataFrame
        Phases that overlap with the frame range
    """
    return phases_of_play[
        (phases_of_play["frame_start"] <= frame_end)
        & (phases_of_play["frame_end"] >= frame_start)
    ].copy()


def get_phase_summary(phases_of_play: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for phases of play.

    Parameters
    ----------
    phases_of_play : pd.DataFrame
        Phases of play DataFrame

    Returns
    -------
    pd.DataFrame
        Summary statistics by phase type
    """
    summary = (
        phases_of_play.groupby(
            ["team_in_possession_phase_type", "team_out_of_possession_phase_type"]
        )
        .agg(
            {
                "index": "count",
                "duration": "sum",
                "team_possession_lead_to_goal": "sum",
                "team_possession_lead_to_shot": "sum",
                "n_player_possessions_in_phase": "sum",
                "team_possession_loss_in_phase": "sum",
            }
        )
        .reset_index()
    )

    summary = summary.rename(
        columns={
            "index": "n_phases",
            "duration": "total_duration",
            "team_possession_lead_to_goal": "n_goals",
            "team_possession_lead_to_shot": "n_shots",
            "n_player_possessions_in_phase": "total_player_possessions",
            "team_possession_loss_in_phase": "total_possession_losses",
        }
    )

    return summary


def load_phases_for_pressing_analysis(
    match_id: int,
    pressing_events: pd.DataFrame,
    data_dir: Optional[Union[str, Path]] = None,
    use_github: bool = False,
) -> pd.DataFrame:
    """
    Load phases of play and link them to pressing events.

    Parameters
    ----------
    match_id : int
        Match ID
    pressing_events : pd.DataFrame
        Pressing events DataFrame
    data_dir : str or Path, optional
        Local directory containing match data
    use_github : bool, default False
        If True, load from GitHub. If False, load from local files.

    Returns
    -------
    pd.DataFrame
        Pressing events with phase context
    """
    phases_of_play = load_phases_of_play(match_id, data_dir, use_github)
    return link_pressing_to_phase(pressing_events, phases_of_play)
