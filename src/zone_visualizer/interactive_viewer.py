"""
Interactive viewer with matplotlib widgets for frame navigation and overlay toggles.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
from .visualizer import ZoneVisualizer
from .zone_calculator import ZoneCalculator, ZoneConfig
from .data_loader import TrackingDataLoader


class InteractiveZoneViewer:
    """Interactive viewer for zone visualization with frame navigation."""

    def __init__(
        self,
        tracking_data: pd.DataFrame,
        match_metadata: Dict,
        dynamic_events: Optional[pd.DataFrame] = None,
        zone_config: Optional[ZoneConfig] = None,
    ):
        """
        Initialize interactive viewer.

        Args:
            tracking_data: Tracking DataFrame
            match_metadata: Match metadata
            dynamic_events: Optional dynamic events DataFrame
            zone_config: Optional zone configuration
        """
        self.tracking_data = tracking_data
        self.match_metadata = match_metadata
        self.dynamic_events = dynamic_events
        self.zone_config = zone_config or ZoneConfig()

        self.zone_calculator = ZoneCalculator(self.zone_config)
        self.data_loader = TrackingDataLoader()
        self.visualizer = ZoneVisualizer(
            pitch_length=match_metadata.get("pitch_length", 105.0),
            pitch_width=match_metadata.get("pitch_width", 68.0),
        )

        # Get frame range
        self.frames = sorted(tracking_data["frame"].unique())
        self.min_frame = min(self.frames)
        self.max_frame = max(self.frames)
        self.current_frame_idx = 0

        # Overlay states
        self.show_overlays = True
        self.show_labels = True
        self.show_zone_boundaries = True
        self.show_trajectories = True
        self.trajectory_frames = 30  # Number of frames to show in trajectory

        # Event mapping
        self.event_frames = self._map_events_to_frames()

        # Setup figure and axes
        self.fig = None
        self.ax = None
        self.pitch = None

        # Widgets
        self.frame_slider = None
        self.overlay_checkboxes = None
        self.event_buttons = {}

    def _map_events_to_frames(self) -> Dict[int, List[Dict]]:
        """Map dynamic events to frames."""
        if self.dynamic_events is None or self.dynamic_events.empty:
            return {}

        event_frames = {}
        for _, event in self.dynamic_events.iterrows():
            frame_start = event.get("frame_start")
            frame_end = event.get("frame_end")

            if pd.notna(frame_start):
                frame_start = int(frame_start)
                if frame_start not in event_frames:
                    event_frames[frame_start] = []
                event_frames[frame_start].append(
                    {
                        "event_id": event.get("event_id"),
                        "event_type": event.get("event_type"),
                        "player_name": event.get("player_name"),
                        "frame_start": frame_start,
                        "frame_end": int(frame_end) if pd.notna(frame_end) else None,
                    }
                )

        return event_frames

    def launch(self) -> None:
        """Launch the interactive viewer."""
        # Create figure and pitch
        self.pitch, self.fig, self.ax = self.visualizer.create_pitch()

        # Adjust layout to make room for widgets
        plt.subplots_adjust(
            left=0.1,
            bottom=0.25,
            right=0.95,
            top=0.95,
        )

        # Create frame slider
        ax_slider = plt.axes([0.1, 0.15, 0.6, 0.03])
        self.frame_slider = Slider(
            ax_slider,
            "Frame",
            self.min_frame,
            self.max_frame,
            valinit=self.frames[0],
            valstep=1,
        )
        self.frame_slider.on_changed(self._update_frame)

        # Create overlay checkboxes
        ax_checkboxes = plt.axes([0.75, 0.05, 0.15, 0.20])
        self.overlay_checkboxes = CheckButtons(
            ax_checkboxes,
            ["Overlays", "Labels", "Boundaries", "Trajectories"],
            [
                self.show_overlays,
                self.show_labels,
                self.show_zone_boundaries,
                self.show_trajectories,
            ],
        )
        self.overlay_checkboxes.on_clicked(self._update_overlays)

        # Create event navigation buttons
        if self.event_frames:
            ax_prev_event = plt.axes([0.1, 0.05, 0.1, 0.04])
            ax_next_event = plt.axes([0.22, 0.05, 0.1, 0.04])
            ax_next_pass = plt.axes([0.34, 0.05, 0.12, 0.04])
            ax_next_shot = plt.axes([0.48, 0.05, 0.12, 0.04])

            self.event_buttons["prev"] = Button(ax_prev_event, "Prev Event")
            self.event_buttons["next"] = Button(ax_next_event, "Next Event")
            self.event_buttons["next_pass"] = Button(ax_next_pass, "Next Pass")
            self.event_buttons["next_shot"] = Button(ax_next_shot, "Next Shot")

            self.event_buttons["prev"].on_clicked(self._prev_event)
            self.event_buttons["next"].on_clicked(self._next_event)
            self.event_buttons["next_pass"].on_clicked(self._next_pass)
            self.event_buttons["next_shot"].on_clicked(self._next_shot)

        # Initial plot
        self._plot_current_frame()

        plt.show()

    def _plot_current_frame(self) -> None:
        """Plot the current frame."""
        if self.ax is None:
            return

        # Clear axes (but keep pitch)
        self.ax.clear()
        self.pitch.draw(self.ax)

        current_frame = self.frames[self.current_frame_idx]

        # Get frame data
        frame_data = self.data_loader.get_frame_data(
            self.tracking_data,
            current_frame,
            self.match_metadata,
        )

        # Determine attacking direction
        period = (
            self.tracking_data[self.tracking_data["frame"] == current_frame][
                "period_id"
            ].iloc[0]
            if not self.tracking_data[
                self.tracking_data["frame"] == current_frame
            ].empty
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
                    current_frame,
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
        event_info = self._get_event_info(current_frame)

        # Get trajectory data for ball carrier (only if enabled)
        trajectory_data = None
        if self.show_trajectories:
            trajectory_data = self._get_trajectory_data(
                current_frame, frame_data["ball_carrier_id"]
            )

        # Plot frame
        self.visualizer.plot_frame(
            ax=self.ax,
            pitch=self.pitch,
            frame_data=frame_data,
            zone_assignments=zone_assignments,
            show_overlays=self.show_overlays,
            show_labels=self.show_labels,
            show_zone_boundaries=self.show_zone_boundaries,
            zone_boundaries=zone_boundaries,
            player_metadata=player_metadata,
            trajectory_data=trajectory_data,
        )

        # Add frame info
        timestamp = (
            self.tracking_data[self.tracking_data["frame"] == current_frame][
                "timestamp"
            ].iloc[0]
            if not self.tracking_data[
                self.tracking_data["frame"] == current_frame
            ].empty
            else None
        )

        self.visualizer.add_frame_info(
            self.ax,
            current_frame,
            timestamp,
            period,
            event_info,
        )

        # Add legend
        self.visualizer.add_legend(self.ax)

        self.fig.canvas.draw()

    def _update_frame(self, val: float) -> None:
        """Update frame when slider changes."""
        frame = int(val)
        if frame in self.frames:
            self.current_frame_idx = self.frames.index(frame)
            self._plot_current_frame()

    def _update_overlays(self, label: str) -> None:
        """Update overlay visibility when checkbox changes."""
        if label == "Overlays":
            self.show_overlays = not self.show_overlays
        elif label == "Labels":
            self.show_labels = not self.show_labels
        elif label == "Boundaries":
            self.show_zone_boundaries = not self.show_zone_boundaries
        elif label == "Trajectories":
            self.show_trajectories = not self.show_trajectories

        self._plot_current_frame()

    def _get_trajectory_data(
        self, frame: int, ball_carrier_id: Optional[int]
    ) -> Optional[Dict]:
        """
        Get trajectory data for ball carrier over previous frames.

        Args:
            frame: Current frame
            ball_carrier_id: Ball carrier player ID

        Returns:
            Dictionary with 'x' and 'y' lists of positions, or None
        """
        if ball_carrier_id is None:
            return None

        # Get frames in trajectory window
        start_frame = max(frame - self.trajectory_frames, self.min_frame)
        trajectory_frames = self.tracking_data[
            (self.tracking_data["frame"] >= start_frame)
            & (self.tracking_data["frame"] <= frame)
            & (self.tracking_data["player_id"] == ball_carrier_id)
        ].sort_values("frame")

        if trajectory_frames.empty:
            return None

        # Extract positions
        x_positions = trajectory_frames["x"].tolist()
        y_positions = trajectory_frames["y"].tolist()

        return {
            "x": x_positions,
            "y": y_positions,
        }

    def _get_event_info(self, frame: int) -> Optional[str]:
        """Get event information for a frame."""
        if frame in self.event_frames:
            events = self.event_frames[frame]
            event_strs = []
            for event in events[:2]:  # Show up to 2 events
                event_type = event.get("event_type", "Event")
                player_name = event.get("player_name", "")
                if player_name:
                    event_strs.append(f"{event_type}: {player_name}")
                else:
                    event_strs.append(event_type)
            return " | ".join(event_strs)
        return None

    def _get_player_metadata(self) -> Dict:
        """Get player metadata from match metadata."""
        player_metadata = {}

        for team_key in ["home_team", "away_team"]:
            if team_key in self.match_metadata:
                team = self.match_metadata[team_key]
                if "players" in team:
                    # Handle both string and int player IDs
                    for player_id, player_info in team["players"].items():
                        # Store with both int and string keys for flexible lookup
                        player_metadata[player_id] = player_info
                        try:
                            if isinstance(player_id, str):
                                player_metadata[int(player_id)] = player_info
                            elif isinstance(player_id, int):
                                player_metadata[str(player_id)] = player_info
                        except (ValueError, TypeError):
                            pass

        return player_metadata

    def _prev_event(self, event) -> None:
        """Navigate to previous event frame."""
        if not self.event_frames:
            return

        event_frames = sorted(self.event_frames.keys())
        current_frame = self.frames[self.current_frame_idx]

        # Find previous event frame that exists in tracking data
        prev_event_frame = None
        for ef in reversed(event_frames):
            if ef < current_frame and ef in self.frames:
                prev_event_frame = ef
                break

        if prev_event_frame is not None:
            self.current_frame_idx = self.frames.index(prev_event_frame)
            self.frame_slider.set_val(prev_event_frame)

    def _next_event(self, event) -> None:
        """Navigate to next event frame."""
        if not self.event_frames:
            return

        event_frames = sorted(self.event_frames.keys())
        current_frame = self.frames[self.current_frame_idx]

        # Find next event frame that exists in tracking data
        next_event_frame = None
        for ef in event_frames:
            if ef > current_frame and ef in self.frames:
                next_event_frame = ef
                break

        if next_event_frame is not None:
            self.current_frame_idx = self.frames.index(next_event_frame)
            self.frame_slider.set_val(next_event_frame)

    def _next_pass(self, event) -> None:
        """Navigate to next pass event."""
        if self.dynamic_events is None:
            return

        passes = self.dynamic_events[self.dynamic_events["event_type"] == "pass"]

        if passes.empty:
            return

        current_frame = self.frames[self.current_frame_idx]
        next_pass = passes[passes["frame_start"] > current_frame]

        if not next_pass.empty:
            next_frame = int(next_pass.iloc[0]["frame_start"])
            # Find closest frame that exists in tracking data
            if next_frame in self.frames:
                self.current_frame_idx = self.frames.index(next_frame)
                self.frame_slider.set_val(next_frame)
            else:
                # Find closest frame
                closest_frame = min(self.frames, key=lambda x: abs(x - next_frame))
                self.current_frame_idx = self.frames.index(closest_frame)
                self.frame_slider.set_val(closest_frame)

    def _next_shot(self, event) -> None:
        """Navigate to next shot event."""
        if self.dynamic_events is None:
            return

        shots = self.dynamic_events[
            self.dynamic_events["event_type"].str.contains("shot", case=False, na=False)
        ]

        if shots.empty:
            return

        current_frame = self.frames[self.current_frame_idx]
        next_shot = shots[shots["frame_start"] > current_frame]

        if not next_shot.empty:
            next_frame = int(next_shot.iloc[0]["frame_start"])
            # Find closest frame that exists in tracking data
            if next_frame in self.frames:
                self.current_frame_idx = self.frames.index(next_frame)
                self.frame_slider.set_val(next_frame)
            else:
                # Find closest frame
                closest_frame = min(self.frames, key=lambda x: abs(x - next_frame))
                self.current_frame_idx = self.frames.index(closest_frame)
                self.frame_slider.set_val(closest_frame)
