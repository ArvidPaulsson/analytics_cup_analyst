"""
Example usage of the pressing data loading utilities.

This script demonstrates how to load and preprocess data for pressing analysis.
"""

from src import (
    load_match_data,
    load_pressing_events,
    create_pressing_dataset,
    load_and_preprocess_tracking,
    calculate_pressing_metrics_by_defensive_phase,
    calculate_pressing_metrics_by_subtype_and_phase,
    get_pressing_effectiveness_score,
    KLOPPY_AVAILABLE,
)

# Example 1: Load all match data
match_id = 1925299
match_data = load_match_data(match_id, use_github=False, load_tracking=False)

print("Match Metadata:")
print(f"Home Team: {match_data['metadata']['home_team']['name']}")
print(f"Away Team: {match_data['metadata']['away_team']['name']}")
print(
    f"Score: {match_data['metadata']['home_team_score']} - {match_data['metadata']['away_team_score']}"
)

# Example 2: Load pressing events
pressing_events = load_pressing_events(match_id, use_github=False)
print(f"\nTotal pressing events: {len(pressing_events)}")
print(f"Pressing events by type:\n{pressing_events['event_subtype'].value_counts()}")

# Example 3: Create unified pressing dataset (without tracking for faster execution)
pressing_dataset = create_pressing_dataset(
    match_id,
    use_github=False,
    load_tracking=False,  # Set to True to include tracking data (slower)
)

print(f"\nPressing events with phases: {len(pressing_dataset['pressing_with_phases'])}")

# Example 4: Analyze pressing by phase type
if len(pressing_dataset["pressing_with_phases"]) > 0:
    pressing_by_phase = (
        pressing_dataset["pressing_with_phases"]
        .groupby("phase_team_in_possession_phase_type")
        .size()
    )
    print(f"\nPressing events by phase type:\n{pressing_by_phase}")

# Example 4b: Calculate comprehensive pressing metrics by defensive phase type
if len(pressing_dataset["pressing_with_phases"]) > 0:
    pressing_metrics = calculate_pressing_metrics_by_defensive_phase(
        pressing_dataset["pressing_with_phases"],
        group_by_team=True,
    )
    print(f"\nPressing metrics by defensive phase type:")
    if len(pressing_metrics) > 0:
        # Select only columns that exist
        available_cols = [
            col
            for col in [
                "phase_team_out_of_possession_phase_type",
                "team_id",
                "team_shortname",
                "total_pressing_events",
                "possession_loss_rate",
                "shot_rate",
                "avg_duration",
            ]
            if col in pressing_metrics.columns
        ]
        if available_cols:
            print(pressing_metrics[available_cols].to_string())
        else:
            print("No matching columns found. Available columns:")
            print(pressing_metrics.columns.tolist())
    else:
        print("No metrics calculated (no data available)")

# Example 4c: Calculate metrics by subtype and phase
if len(pressing_dataset["pressing_with_phases"]) > 0:
    subtype_metrics = calculate_pressing_metrics_by_subtype_and_phase(
        pressing_dataset["pressing_with_phases"]
    )
    print(f"\nPressing metrics by subtype and defensive phase:")
    print(
        subtype_metrics[
            [
                "event_subtype",
                "phase_team_out_of_possession_phase_type",
                "total_events",
                "possession_loss_rate",
            ]
        ]
        .head(10)
        .to_string()
    )

# Example 4d: Calculate pressing effectiveness score
if len(pressing_dataset["pressing_with_phases"]) > 0:
    effectiveness = get_pressing_effectiveness_score(pressing_metrics)
    print(f"\nPressing effectiveness by defensive phase type:")
    print(
        effectiveness[
            [
                "phase_team_out_of_possession_phase_type",
                "pressing_effectiveness_score",
                "possession_loss_rate",
                "danger_stopped_rate",
            ]
        ].to_string()
    )

# Example 5: Load and preprocess tracking data (Kloppy only)
if not KLOPPY_AVAILABLE:
    raise ImportError(
        "Kloppy is required for tracking operations. Install it with `pip install kloppy>=3.18.0`."
    )

tracking_data = load_and_preprocess_tracking(
    match_id,
    use_github=True,
    sample_rate=0.5,  # downsample for speed
    limit=200,  # limit frames for the demo
)
print(f"\nTracking frames (kloppy): {tracking_data['frame'].nunique()}")

print("\nExample usage completed!")
