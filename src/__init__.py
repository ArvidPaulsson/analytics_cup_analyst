"""
Pressing Efficiency and Evasion Analysis - Data Loading Utilities

This package provides utilities for loading and preprocessing data for pressing analysis,
including dynamic events, phases of play, and tracking data.
"""

from .data_loader import (
    load_match_metadata,
    load_dynamic_events,
    load_phases_of_play,
    load_tracking_data,
    load_tracking_dataset_kloppy,
    KLOPPY_AVAILABLE,
    load_match_data,
    load_all_matches_metadata,
)

from .pressing_data_loader import (
    extract_pressing_events,
    load_pressing_events,
    extract_pressing_chains,
    get_pressing_chain_summary,
    filter_pressing_by_phase_type,
    get_pressing_event_outcomes,
    get_pressing_by_location,
)

from .tracking_loader import (
    preprocess_tracking_data,
    extract_frame_data,
    extract_frame_range,
    get_player_positions,
    get_ball_position,
    get_team_positions,
    calculate_player_distances,
    load_and_preprocess_tracking,
    get_tracking_for_event,
)

from .phase_loader import (
    link_pressing_to_phase,
    get_phases_by_type,
    get_phases_by_frame_range,
    get_phase_summary,
    load_phases_for_pressing_analysis,
)

from .data_merger import (
    merge_pressing_with_tracking,
    merge_pressing_with_phases,
    create_pressing_situation_dataset,
    create_pressing_dataset,
    get_pressing_event_tracking_snapshot,
    aggregate_pressing_by_frame,
    get_pressing_event_summary,
)

from .data_preprocessing import (
    normalize_coordinates,
    handle_missing_data,
    calculate_distance,
    calculate_angle,
    calculate_speed,
    calculate_derived_metrics,
    add_pitch_zones,
    preprocess_pressing_data,
    get_pressing_intensity,
)

from .pressing_metrics import (
    calculate_pressing_metrics_by_defensive_phase,
    calculate_pressing_metrics_by_subtype_and_phase,
    get_pressing_effectiveness_score,
)

__all__ = [
    # Data loader
    "load_match_metadata",
    "load_dynamic_events",
    "load_phases_of_play",
    "load_tracking_data",
    "load_tracking_dataset_kloppy",
    "KLOPPY_AVAILABLE",
    "load_match_data",
    "load_all_matches_metadata",
    # Pressing data loader
    "extract_pressing_events",
    "load_pressing_events",
    "extract_pressing_chains",
    "get_pressing_chain_summary",
    "filter_pressing_by_phase_type",
    "get_pressing_event_outcomes",
    "get_pressing_by_location",
    # Tracking loader
    "preprocess_tracking_data",
    "extract_frame_data",
    "extract_frame_range",
    "get_player_positions",
    "get_ball_position",
    "get_team_positions",
    "calculate_player_distances",
    "load_and_preprocess_tracking",
    "get_tracking_for_event",
    # Phase loader
    "link_pressing_to_phase",
    "get_phases_by_type",
    "get_phases_by_frame_range",
    "get_phase_summary",
    "load_phases_for_pressing_analysis",
    # Data merger
    "merge_pressing_with_tracking",
    "merge_pressing_with_phases",
    "create_pressing_situation_dataset",
    "create_pressing_dataset",
    "get_pressing_event_tracking_snapshot",
    "aggregate_pressing_by_frame",
    "get_pressing_event_summary",
    # Data preprocessing
    "normalize_coordinates",
    "handle_missing_data",
    "calculate_distance",
    "calculate_angle",
    "calculate_speed",
    "calculate_derived_metrics",
    "add_pitch_zones",
    "preprocess_pressing_data",
    "get_pressing_intensity",
    # Pressing metrics
    "calculate_pressing_metrics_by_defensive_phase",
    "calculate_pressing_metrics_by_subtype_and_phase",
    "get_pressing_effectiveness_score",
]
