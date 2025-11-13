"""
Plotly-based interactive viewer with ipywidgets for Jupyter notebooks.

This module provides an interactive viewer using Plotly and ipywidgets,
optimized for Jupyter notebook environments.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
from .visualizer_plotly import ZoneVisualizerPlotly
from .zone_calculator import ZoneCalculator, ZoneConfig
from .data_loader import TrackingDataLoader


class InteractiveZoneViewerPlotly:
    """Interactive viewer for zone visualization using Plotly and ipywidgets."""

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
        self.visualizer = ZoneVisualizerPlotly(
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
        self.trajectory_frames = 30

        # Event mapping
        self.event_frames = self._map_events_to_frames()

        # Widgets and figure
        self.fig = None
        self.output = None
        self._setup_widgets()

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

    def _setup_widgets(self) -> None:
        """Setup ipywidgets for interaction."""
        # Frame slider
        self.frame_slider = widgets.IntSlider(
            value=self.frames[0],
            min=self.min_frame,
            max=self.max_frame,
            step=1,
            description="Frame:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="600px"),
        )
        self.frame_slider.observe(self._update_frame, names="value")

        # Overlay checkboxes
        self.overlay_checkbox = widgets.Checkbox(
            value=self.show_overlays,
            description="Show Overlays",
            indent=False,
        )
        self.overlay_checkbox.observe(self._update_overlay, names="value")

        self.labels_checkbox = widgets.Checkbox(
            value=self.show_labels,
            description="Show Labels",
            indent=False,
        )
        self.labels_checkbox.observe(self._update_labels, names="value")

        self.boundaries_checkbox = widgets.Checkbox(
            value=self.show_zone_boundaries,
            description="Show Boundaries",
            indent=False,
        )
        self.boundaries_checkbox.observe(self._update_boundaries, names="value")

        self.trajectories_checkbox = widgets.Checkbox(
            value=self.show_trajectories,
            description="Show Trajectories",
            indent=False,
        )
        self.trajectories_checkbox.observe(self._update_trajectories, names="value")

        # Event navigation buttons
        self.prev_event_btn = widgets.Button(
            description="⏮ Prev Event",
            button_style="",
            layout=widgets.Layout(width="120px"),
        )
        self.prev_event_btn.on_click(self._prev_event)

        self.next_event_btn = widgets.Button(
            description="Next Event ⏭",
            button_style="",
            layout=widgets.Layout(width="120px"),
        )
        self.next_event_btn.on_click(self._next_event)

        self.next_pass_btn = widgets.Button(
            description="Next Pass ⚽",
            button_style="",
            layout=widgets.Layout(width="120px"),
        )
        self.next_pass_btn.on_click(self._next_pass)

        self.next_shot_btn = widgets.Button(
            description="Next Shot 🎯",
            button_style="",
            layout=widgets.Layout(width="120px"),
        )
        self.next_shot_btn.on_click(self._next_shot)

        # Output widget for figure
        self.output = widgets.Output()

    def launch(self) -> None:
        """Launch the interactive viewer in Jupyter notebook."""
        # Create initial figure
        self._plot_current_frame()

        # Layout widgets
        controls_box = widgets.VBox([
            widgets.HBox([
                self.frame_slider,
            ]),
            widgets.HBox([
                self.overlay_checkbox,
                self.labels_checkbox,
                self.boundaries_checkbox,
                self.trajectories_checkbox,
            ]),
            widgets.HBox([
                self.prev_event_btn,
                self.next_event_btn,
                self.next_pass_btn,
                self.next_shot_btn,
            ]) if self.event_frames else widgets.HBox([]),
        ])

        # Display everything
        display(controls_box, self.output)

    def _plot_current_frame(self) -> None:
        """Plot the current frame."""
        with self.output:
            self.output.clear_output(wait=True)

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

                # Always try to calculate from position differences as fallback
                if ball_carrier_pos is not None:
                    movement_data = self.data_loader.get_ball_carrier_movement(
                        self.tracking_data,
                        current_frame,
                        lookback_frames=5,
                    )
                    if movement_data and movement_data.get("direction") is not None:
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

            # Create figure
            timestamp = (
                self.tracking_data[self.tracking_data["frame"] == current_frame][
                    "timestamp"
                ].iloc[0]
                if not self.tracking_data[
                    self.tracking_data["frame"] == current_frame
                ].empty
                else None
            )

            self.fig = self.visualizer.create_pitch_figure(
                title=f"Frame {current_frame}",
            )

            # Plot frame
            self.visualizer.plot_frame(
                fig=self.fig,
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
            self.visualizer.add_frame_info(
                self.fig,
                current_frame,
                timestamp,
                period,
                event_info,
            )

            # Display figure
            self.fig.show()

    def _update_frame(self, change: Dict) -> None:
        """Update frame when slider changes."""
        frame = change["new"]
        if frame in self.frames:
            self.current_frame_idx = self.frames.index(frame)
            self._plot_current_frame()

    def _update_overlay(self, change: Dict) -> None:
        """Update overlay visibility."""
        self.show_overlays = change["new"]
        self._plot_current_frame()

    def _update_labels(self, change: Dict) -> None:
        """Update labels visibility."""
        self.show_labels = change["new"]
        self._plot_current_frame()

    def _update_boundaries(self, change: Dict) -> None:
        """Update boundaries visibility."""
        self.show_zone_boundaries = change["new"]
        self._plot_current_frame()

    def _update_trajectories(self, change: Dict) -> None:
        """Update trajectories visibility."""
        self.show_trajectories = change["new"]
        self._plot_current_frame()

    def _get_trajectory_data(
        self, frame: int, ball_carrier_id: Optional[int]
    ) -> Optional[Dict]:
        """Get trajectory data for ball carrier over previous frames."""
        if ball_carrier_id is None:
            return None

        start_frame = max(frame - self.trajectory_frames, self.min_frame)
        trajectory_frames = self.tracking_data[
            (self.tracking_data["frame"] >= start_frame)
            & (self.tracking_data["frame"] <= frame)
            & (self.tracking_data["player_id"] == ball_carrier_id)
        ].sort_values("frame")

        if trajectory_frames.empty:
            return None

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
            for event in events[:2]:
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
                    for player_id, player_info in team["players"].items():
                        player_metadata[player_id] = player_info
                        try:
                            if isinstance(player_id, str):
                                player_metadata[int(player_id)] = player_info
                            elif isinstance(player_id, int):
                                player_metadata[str(player_id)] = player_info
                        except (ValueError, TypeError):
                            pass

        return player_metadata

    def _prev_event(self, button: widgets.Button) -> None:
        """Navigate to previous event frame."""
        if not self.event_frames:
            return

        event_frames = sorted(self.event_frames.keys())
        current_frame = self.frames[self.current_frame_idx]

        prev_event_frame = None
        for ef in reversed(event_frames):
            if ef < current_frame and ef in self.frames:
                prev_event_frame = ef
                break

        if prev_event_frame is not None:
            self.current_frame_idx = self.frames.index(prev_event_frame)
            self.frame_slider.value = prev_event_frame

    def _next_event(self, button: widgets.Button) -> None:
        """Navigate to next event frame."""
        if not self.event_frames:
            return

        event_frames = sorted(self.event_frames.keys())
        current_frame = self.frames[self.current_frame_idx]

        next_event_frame = None
        for ef in event_frames:
            if ef > current_frame and ef in self.frames:
                next_event_frame = ef
                break

        if next_event_frame is not None:
            self.current_frame_idx = self.frames.index(next_event_frame)
            self.frame_slider.value = next_event_frame

    def _next_pass(self, button: widgets.Button) -> None:
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
            if next_frame in self.frames:
                self.current_frame_idx = self.frames.index(next_frame)
                self.frame_slider.value = next_frame
            else:
                closest_frame = min(self.frames, key=lambda x: abs(x - next_frame))
                self.current_frame_idx = self.frames.index(closest_frame)
                self.frame_slider.value = closest_frame

    def _next_shot(self, button: widgets.Button) -> None:
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
            if next_frame in self.frames:
                self.current_frame_idx = self.frames.index(next_frame)
                self.frame_slider.value = next_frame
            else:
                closest_frame = min(self.frames, key=lambda x: abs(x - next_frame))
                self.current_frame_idx = self.frames.index(closest_frame)
                self.frame_slider.value = closest_frame

