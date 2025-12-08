"""
Movement DNA Module

Extract match-level features and create embeddings for midfielder movement analysis.
"""

from .event_features import (
    extract_event_features,
    extract_event_features_from_file,
    extract_event_features_from_multiple_matches,
    extract_obr_events,
    extract_po_events,
    extract_pp_events,
    extract_obe_events,
    extract_events_by_type,
)

from .physical_features import (
    extract_physical_features,
    load_and_extract_physical_features,
    get_physical_features_for_clustering,
)

from .tracking_features import (
    extract_tracking_features,
    extract_tracking_features_from_file,
    extract_tracking_features_from_kloppy,
    load_tracking_with_kloppy,
)

from .phase_features import (
    extract_phase_context_features,
    load_and_extract_phase_features,
    extract_phase_features_from_multiple_matches,
    get_phase_for_frame,
)


__all__ = [
    "extract_event_features",
    "extract_event_features_from_file",
    "extract_event_features_from_multiple_matches",
    "extract_obr_events",
    "extract_po_events",
    "extract_pp_events",
    "extract_obe_events",
    "extract_events_by_type",
    "extract_physical_features",
    "load_and_extract_physical_features",
    "get_physical_features_for_clustering",
    "extract_tracking_features",
    "extract_tracking_features_from_file",
    "extract_tracking_features_from_kloppy",
    "load_tracking_with_kloppy",
    "extract_phase_context_features",
    "load_and_extract_phase_features",
    "extract_phase_features_from_multiple_matches",
    "get_phase_for_frame",
    "calculate_mis_per_player_match",
    "calculate_mis_for_reception",
    "calculate_acceleration_and_hsr_before_receiving",
    "load_and_calculate_mis",
]
