"""
Example script demonstrating zone visualizer usage.

This script shows how to:
1. Load match data
2. Configure zone parameters
3. Visualize frames
4. Launch interactive viewer

Note: Zone 1 (support zone) automatically orients itself based on the ball carrier's
movement direction. The zone_1_orientation parameter is only used as a fallback when
movement direction data is not available.
"""

from .zone_layer import ZoneLayer
from .zone_calculator import ZoneConfig


def main():
    """Main example function."""
    # Match ID from the dataset
    match_id = 1886347

    # Create zone configuration
    # You can customize these parameters:
    zone_config = ZoneConfig(
        zone_0_radius=5.0,  # Zone 0 is just the ball carrier
        zone_1_radius=15.0,  # Support zone radius (meters)
        zone_1_angle=60.0,  # Angular cone for Zone 1 (±60 degrees)
        zone_1_orientation="toward_goal",  # Fallback orientation (ball movement direction takes priority)
        zone_2_radius=None,  # None means all remaining players
    )

    print(f"Loading match {match_id}...")

    # Initialize zone layer
    # This will automatically load tracking data and events
    zone_layer = ZoneLayer(
        match_id=match_id,
        zone_config=zone_config,
        use_kloppy=True,  # Use kloppy if available
    )

    print("Data loaded successfully!")
    print(f"Total frames: {len(zone_layer.tracking_data['frame'].unique())}")

    # Example 1: Visualize a specific frame
    print("\nExample 1: Visualizing a specific frame...")
    example_frame = 100  # Choose a frame number
    zone_layer.visualize_frame(
        frame=example_frame,
        show_overlays=True,
        show_labels=True,
        show_zone_boundaries=True,
    )

    # Example 2: Get zone statistics for a frame
    print("\nExample 2: Zone statistics for a frame...")
    stats = zone_layer.get_zone_stats(example_frame)
    print(f"Frame {stats['frame']}:")
    print(f"  Ball carrier: {stats['ball_carrier_id']}")
    print(f"  Zone 0 (Ball carrier): {stats['zone_counts'][0]}")
    print(f"  Zone 1 (Support): {stats['zone_counts'][1]}")
    print(f"  Zone 2 (Cooperation): {stats['zone_counts'][2]}")
    print(f"  Total players: {stats['total_players']}")

    # Example 3: Launch interactive viewer
    print("\nExample 3: Launching interactive viewer...")
    print("Use the slider to navigate frames, checkboxes to toggle overlays,")
    print("and buttons to jump to events.")
    zone_layer.launch_interactive()

    # Example 4: Custom zone configuration
    print("\nExample 4: Using custom zone configuration...")
    custom_config = ZoneConfig(
        zone_1_radius=20.0,  # Larger support zone
        zone_1_angle=90.0,  # Wider angle cone
        zone_1_orientation="toward_goal",  # Fallback (ball movement direction used when available)
    )
    zone_layer.update_zone_config(custom_config)
    print("Zone configuration updated!")
    print("Note: Zone 1 angle automatically follows ball carrier movement direction.")
    print(
        "The 'toward_goal' orientation is only used as a fallback when movement data is unavailable."
    )


if __name__ == "__main__":
    main()
