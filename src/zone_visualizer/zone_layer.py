"""
Main interface for zone layer visualization.

Combines data loading, zone calculation, and visualization into a single interface.
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd
from pathlib import Path

from .data_loader import TrackingDataLoader
from .zone_calculator import ZoneConfig, ZoneCalculator
from .visualizer import ZoneVisualizer
from .interactive_viewer import InteractiveZoneViewer


class ZoneLayer:
    """Main interface for zone visualization."""

    def __init__(
        self,
        match_id: int,
        data_dir: Optional[Path] = None,
        zone_config: Optional[ZoneConfig] = None,
        use_kloppy: Optional[bool] = None,
    ):
        """
        Initialize zone layer for a match.

        Args:
            match_id: Match ID to load
            data_dir: Optional base directory for data files
            zone_config: Optional zone configuration
            use_kloppy: Whether to use kloppy (default: True if available)
        """
        self.match_id = match_id
        self.data_dir = data_dir
        self.zone_config = zone_config or ZoneConfig()

        # Initialize components
        self.data_loader = TrackingDataLoader(data_dir)
        self.zone_calculator = ZoneCalculator(self.zone_config)

        # Load data
        self._load_data(use_kloppy)

        # Initialize visualizer
        pitch_length = self.match_metadata.get("pitch_length", 105.0)
        pitch_width = self.match_metadata.get("pitch_width", 68.0)
        self.visualizer = ZoneVisualizer(
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

    def _load_data(self, use_kloppy: Optional[bool] = None) -> None:
        """Load tracking and event data."""
        # Load tracking data
        tracking_result = self.data_loader.load_match_data(
            self.match_id,
            use_kloppy=use_kloppy,
        )

        self.tracking_data = tracking_result["tracking_data"]
        self.match_metadata = tracking_result["match_metadata"]

        # Try to load dynamic events
        try:
            self.dynamic_events = self.data_loader.load_dynamic_events(self.match_id)
        except FileNotFoundError:
            self.dynamic_events = None

    def visualize_frame(
        self,
        frame: int,
        show_overlays: bool = True,
        show_labels: bool = True,
        show_zone_boundaries: bool = True,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Visualize a single frame (static plot).

        Args:
            frame: Frame number to visualize
            show_overlays: Whether to show zone overlays
            show_labels: Whether to show player labels
            show_zone_boundaries: Whether to show zone boundaries
            save_path: Optional path to save the figure
        """
        import matplotlib.pyplot as plt

        # Get frame data
        frame_data = self.data_loader.get_frame_data(
            self.tracking_data,
            frame,
            self.match_metadata,
        )

        # Determine attacking direction
        period = (
            self.tracking_data[self.tracking_data["frame"] == frame]["period_id"].iloc[
                0
            ]
            if not self.tracking_data[self.tracking_data["frame"] == frame].empty
            else 1
        )

        attacking_direction = self.data_loader.get_attacking_direction(
            self.match_metadata,
            period,
        )

        # Get ball carrier movement direction
        ball_carrier_movement_dir = None
        ball_carrier_pos = None
        if frame_data["ball_carrier_id"]:
            for player in frame_data["players"]:
                if player["player_id"] == frame_data["ball_carrier_id"]:
                    ball_carrier_pos = (player["x"], player["y"])
                    # Get movement direction from player data (velocity from kloppy)
                    if player.get("direction") is not None:
                        direction = player["direction"]
                        # Convert from degrees to radians if needed
                        if abs(direction) > 2 * np.pi:
                            ball_carrier_movement_dir = np.deg2rad(direction)
                        else:
                            ball_carrier_movement_dir = direction
                    break

            # Always try to calculate from position differences as fallback or verification
            # This ensures we have movement direction even when velocity data is missing
            if ball_carrier_pos is not None:
                movement_data = self.data_loader.get_ball_carrier_movement(
                    self.tracking_data,
                    frame,
                    lookback_frames=5,
                )
                if movement_data and movement_data.get("direction") is not None:
                    # Use calculated direction (more reliable for visualization)
                    # Velocity data might be None or inaccurate
                    ball_carrier_movement_dir = movement_data["direction"]

        # Calculate zones
        zone_assignments = self.zone_calculator.calculate_zones(
            ball_position=frame_data["ball_position"],
            ball_carrier_id=frame_data["ball_carrier_id"],
            players=frame_data["players"],
            attacking_direction=attacking_direction,
            ball_movement_direction=ball_carrier_movement_dir,
        )

        # Get zone boundaries
        zone_boundaries = None
        if ball_carrier_pos:
            zone_boundaries = self.zone_calculator.get_zone_boundaries(
                ball_carrier_pos,
                attacking_direction,
                ball_movement_direction=ball_carrier_movement_dir,
            )

        # Get player metadata
        player_metadata = self._get_player_metadata()

        # Get event info
        event_info = self._get_event_info(frame)

        # Get trajectory data for ball carrier
        trajectory_data = None
        if show_overlays and frame_data["ball_carrier_id"]:
            trajectory_data = self._get_trajectory_data(
                frame, frame_data["ball_carrier_id"]
            )

        # Create plot
        pitch, fig, ax = self.visualizer.create_pitch()

        # Plot frame
        self.visualizer.plot_frame(
            ax=ax,
            pitch=pitch,
            frame_data=frame_data,
            zone_assignments=zone_assignments,
            show_overlays=show_overlays,
            show_labels=show_labels,
            show_zone_boundaries=show_zone_boundaries,
            zone_boundaries=zone_boundaries,
            player_metadata=player_metadata,
            trajectory_data=trajectory_data,
        )

        # Add frame info
        timestamp = (
            self.tracking_data[self.tracking_data["frame"] == frame]["timestamp"].iloc[
                0
            ]
            if not self.tracking_data[self.tracking_data["frame"] == frame].empty
            else None
        )

        self.visualizer.add_frame_info(
            ax,
            frame,
            timestamp,
            period,
            event_info,
        )

        # Add legend
        self.visualizer.add_legend(ax)

        # Save or show
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def launch_interactive(self) -> None:
        """Launch interactive viewer."""
        viewer = InteractiveZoneViewer(
            tracking_data=self.tracking_data,
            match_metadata=self.match_metadata,
            dynamic_events=self.dynamic_events,
            zone_config=self.zone_config,
        )
        viewer.launch()

    def _get_player_metadata(self) -> Dict:
        """Get player metadata from match metadata."""
        player_metadata = {}

        for team_key in ["home_team", "away_team"]:
            if team_key in self.match_metadata:
                team = self.match_metadata[team_key]
                if "players" in team:
                    # Handle both string and int player IDs
                    for player_id, player_info in team["players"].items():
                        # Store with original key
                        player_metadata[player_id] = player_info
                        # Also store with alternative type for flexible lookup
                        try:
                            if isinstance(player_id, str):
                                player_metadata[int(player_id)] = player_info
                            elif isinstance(player_id, int):
                                player_metadata[str(player_id)] = player_info
                        except (ValueError, TypeError):
                            pass

        return player_metadata

    def _get_event_info(self, frame: int) -> Optional[str]:
        """Get event information for a frame."""
        if self.dynamic_events is None:
            return None

        frame_events = self.dynamic_events[self.dynamic_events["frame_start"] == frame]

        if frame_events.empty:
            return None

        event_strs = []
        for _, event in frame_events.head(2).iterrows():
            event_type = event.get("event_type", "Event")
            player_name = event.get("player_name", "")
            if player_name:
                event_strs.append(f"{event_type}: {player_name}")
            else:
                event_strs.append(event_type)

        return " | ".join(event_strs)

    def update_zone_config(self, config: ZoneConfig) -> None:
        """Update zone configuration."""
        self.zone_config = config
        self.zone_calculator = ZoneCalculator(config)

    def get_zone_stats(self, frame: int) -> Dict:
        """
        Get zone statistics for a frame.

        Returns:
            Dictionary with zone counts and other statistics
        """
        frame_data = self.data_loader.get_frame_data(
            self.tracking_data,
            frame,
            self.match_metadata,
        )

        period = (
            self.tracking_data[self.tracking_data["frame"] == frame]["period_id"].iloc[
                0
            ]
            if not self.tracking_data[self.tracking_data["frame"] == frame].empty
            else 1
        )

        attacking_direction = self.data_loader.get_attacking_direction(
            self.match_metadata,
            period,
        )

        zone_assignments = self.zone_calculator.calculate_zones(
            ball_position=frame_data["ball_position"],
            ball_carrier_id=frame_data["ball_carrier_id"],
            players=frame_data["players"],
            attacking_direction=attacking_direction,
        )

        # Count players per zone
        zone_counts = {0: 0, 1: 0, 2: 0}
        for zone in zone_assignments.values():
            if zone in zone_counts:
                zone_counts[zone] += 1

        return {
            "frame": frame,
            "ball_carrier_id": frame_data["ball_carrier_id"],
            "zone_counts": zone_counts,
            "total_players": len(frame_data["players"]),
        }
