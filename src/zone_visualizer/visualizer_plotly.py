"""
Plotly-based visualization module for zone overlays on football pitch.

This module provides Plotly-based visualization for Jupyter notebooks,
replacing matplotlib/mplsoccer with Plotly for better interactivity.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go


class ZoneVisualizerPlotly:
    """Visualize zones on Plotly football pitch."""

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        """
        Initialize visualizer.

        Args:
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Color scheme
        self.colors = {
            "zone_0": "#FFD700",  # Gold
            "zone_1": "#00CED1",  # Dark turquoise
            "zone_2": "#4169E1",  # Royal blue
            "opponent": "#DC143C",  # Crimson
            "ball": "#FFFFFF",  # White
            "pitch": "#001400",  # Dark green
            "line": "#FFFFFF",  # White
        }

        # Zone labels
        self.zone_labels = {
            0: "Zone 0",
            1: "Zone 1",
            2: "Zone 2",
        }

    def create_pitch_figure(
        self,
        title: Optional[str] = None,
        height: int = 800,
    ) -> go.Figure:
        """
        Create Plotly figure with football pitch.

        Args:
            title: Optional figure title
            height: Figure height in pixels

        Returns:
            Plotly figure with pitch drawn
        """
        fig = go.Figure()

        # Calculate aspect ratio
        aspect_ratio = self.pitch_length / self.pitch_width
        width = int(height * aspect_ratio)

        # Draw pitch background (rectangle)
        fig.add_shape(
            type="rect",
            x0=-self.pitch_length / 2,
            y0=-self.pitch_width / 2,
            x1=self.pitch_length / 2,
            y1=self.pitch_width / 2,
            fillcolor=self.colors["pitch"],
            line=dict(color=self.colors["line"], width=2),
            layer="below",
        )

        # Draw center line
        fig.add_shape(
            type="line",
            x0=0,
            y0=-self.pitch_width / 2,
            x1=0,
            y1=self.pitch_width / 2,
            line=dict(color=self.colors["line"], width=2),
            layer="below",
        )

        # Draw center circle
        center_circle_radius = 9.15
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = center_circle_radius * np.cos(theta)
        circle_y = center_circle_radius * np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                line=dict(color=self.colors["line"], width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Draw penalty areas
        penalty_area_length = 16.5
        penalty_area_width = 40.32
        penalty_arc_radius = 9.15

        # Left penalty area
        fig.add_shape(
            type="rect",
            x0=-self.pitch_length / 2,
            y0=-penalty_area_width / 2,
            x1=-self.pitch_length / 2 + penalty_area_length,
            y1=penalty_area_width / 2,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=self.colors["line"], width=2),
            layer="below",
        )

        # Right penalty area
        fig.add_shape(
            type="rect",
            x0=self.pitch_length / 2 - penalty_area_length,
            y0=-penalty_area_width / 2,
            x1=self.pitch_length / 2,
            y1=penalty_area_width / 2,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=self.colors["line"], width=2),
            layer="below",
        )

        # Draw penalty arcs (semicircles)
        # Left arc
        arc_theta = np.linspace(-np.pi / 2, np.pi / 2, 50)
        arc_x_left = -self.pitch_length / 2 + penalty_area_length + penalty_arc_radius * np.cos(arc_theta)
        arc_y_left = penalty_arc_radius * np.sin(arc_theta)
        fig.add_trace(
            go.Scatter(
                x=arc_x_left,
                y=arc_y_left,
                mode="lines",
                line=dict(color=self.colors["line"], width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Right arc
        arc_x_right = self.pitch_length / 2 - penalty_area_length - penalty_arc_radius * np.cos(arc_theta)
        arc_y_right = penalty_arc_radius * np.sin(arc_theta)
        fig.add_trace(
            go.Scatter(
                x=arc_x_right,
                y=arc_y_right,
                mode="lines",
                line=dict(color=self.colors["line"], width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Draw goals
        goal_width = 7.32
        goal_depth = 2.0

        # Left goal
        fig.add_shape(
            type="rect",
            x0=-self.pitch_length / 2 - goal_depth,
            y0=-goal_width / 2,
            x1=-self.pitch_length / 2,
            y1=goal_width / 2,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=self.colors["line"], width=3),
            layer="below",
        )

        # Right goal
        fig.add_shape(
            type="rect",
            x0=self.pitch_length / 2,
            y0=-goal_width / 2,
            x1=self.pitch_length / 2 + goal_depth,
            y1=goal_width / 2,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=self.colors["line"], width=3),
            layer="below",
        )

        # Draw 6-yard boxes
        six_yard_length = 5.5
        six_yard_width = 18.32

        # Left 6-yard box
        fig.add_shape(
            type="rect",
            x0=-self.pitch_length / 2,
            y0=-six_yard_width / 2,
            x1=-self.pitch_length / 2 + six_yard_length,
            y1=six_yard_width / 2,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=self.colors["line"], width=2),
            layer="below",
        )

        # Right 6-yard box
        fig.add_shape(
            type="rect",
            x0=self.pitch_length / 2 - six_yard_length,
            y0=-six_yard_width / 2,
            x1=self.pitch_length / 2,
            y1=six_yard_width / 2,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=self.colors["line"], width=2),
            layer="below",
        )

        # Update layout
        fig.update_layout(
            title=title or "Zone Visualization",
            xaxis=dict(
                range=[-self.pitch_length / 2 - 5, self.pitch_length / 2 + 5],
                scaleanchor="y",
                scaleratio=1,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                range=[-self.pitch_width / 2 - 5, self.pitch_width / 2 + 5],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            width=width,
            height=height,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="white",
            showlegend=True,
        )

        return fig

    def plot_frame(
        self,
        fig: go.Figure,
        frame_data: Dict,
        zone_assignments: Dict[int, int],
        show_overlays: bool = True,
        show_labels: bool = True,
        show_zone_boundaries: bool = True,
        zone_boundaries: Optional[Dict] = None,
        player_metadata: Optional[Dict] = None,
        trajectory_data: Optional[Dict] = None,
    ) -> None:
        """
        Plot a single frame with zone overlays on Plotly figure.

        Args:
            fig: Plotly figure (will be modified in place)
            frame_data: Frame data dict with ball_position, ball_carrier_id, players
            zone_assignments: Dict mapping player_id -> zone
            show_overlays: Whether to show zone overlays
            show_labels: Whether to show zone labels
            show_zone_boundaries: Whether to show zone boundaries
            zone_boundaries: Optional zone boundary data from ZoneCalculator
            player_metadata: Optional dict mapping player_id -> {name, jersey_no, team_id}
            trajectory_data: Optional trajectory data for ball carrier
        """
        ball_position = frame_data["ball_position"]
        ball_carrier_id = frame_data["ball_carrier_id"]
        players = frame_data["players"]

        # Get possession team from frame data
        possession_team_id = frame_data.get("ball_owning_team_id")

        # If not available, try to get from ball carrier
        if possession_team_id is None and ball_carrier_id:
            for player in players:
                if player["player_id"] == ball_carrier_id:
                    possession_team_id = player.get("team_id")
                    break

        # If we still don't have team, try to infer from zone assignments
        if possession_team_id is None:
            for player in players:
                player_id = player["player_id"]
                zone = zone_assignments.get(player_id, -1)
                if zone in [0, 1, 2]:
                    possession_team_id = player.get("team_id")
                    if possession_team_id is not None:
                        break

        # Plot trajectory path if enabled and available
        if show_overlays and trajectory_data:
            self._plot_trajectory(fig, trajectory_data)

        # Plot ball
        fig.add_trace(
            go.Scatter(
                x=[ball_position[0]],
                y=[ball_position[1]],
                mode="markers",
                marker=dict(
                    size=15,
                    color=self.colors["ball"],
                    line=dict(color="black", width=2),
                ),
                name="Ball",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Group players by zone and team
        players_by_zone = {0: [], 1: [], 2: [], -1: []}
        teammates = []
        opponents = []

        for player in players:
            player_id = player["player_id"]
            zone = zone_assignments.get(player_id, -1)
            player_team_id = player.get("team_id")

            # Determine if teammate or opponent
            if possession_team_id is not None and player_team_id is not None:
                is_teammate = player_team_id == possession_team_id
            else:
                is_teammate = zone in [0, 1, 2]

            # Assign to zone if teammate and in a valid zone
            if is_teammate and zone in [0, 1, 2]:
                players_by_zone[zone].append(player)
            else:
                opponents.append(player)

        # Plot zone boundaries if enabled
        if show_overlays and show_zone_boundaries and zone_boundaries:
            self._plot_zone_boundaries(fig, zone_boundaries)

        # Plot opponents (square markers)
        if opponents:
            opp_x = [p["x"] for p in opponents]
            opp_y = [p["y"] for p in opponents]
            fig.add_trace(
                go.Scatter(
                    x=opp_x,
                    y=opp_y,
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=self.colors["opponent"],
                        line=dict(color="white", width=2),
                        symbol="square",
                    ),
                    name="Opponents",
                    showlegend=True,
                    hoverinfo="text",
                    hovertext=[self._get_player_label(p, player_metadata) for p in opponents],
                )
            )

        # Plot Zone 2 players (cooperation zone)
        if players_by_zone[2]:
            z2_x = [p["x"] for p in players_by_zone[2]]
            z2_y = [p["y"] for p in players_by_zone[2]]
            fig.add_trace(
                go.Scatter(
                    x=z2_x,
                    y=z2_y,
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=self.colors["zone_2"],
                        line=dict(color="white", width=1.5),
                    ),
                    name="Zone 2 (Cooperation)",
                    showlegend=True,
                    hoverinfo="text",
                    hovertext=[self._get_player_label(p, player_metadata) for p in players_by_zone[2]],
                )
            )

        # Plot Zone 1 players (support zone)
        if players_by_zone[1]:
            z1_x = [p["x"] for p in players_by_zone[1]]
            z1_y = [p["y"] for p in players_by_zone[1]]
            fig.add_trace(
                go.Scatter(
                    x=z1_x,
                    y=z1_y,
                    mode="markers",
                    marker=dict(
                        size=14,
                        color=self.colors["zone_1"],
                        line=dict(color="white", width=2),
                    ),
                    name="Zone 1 (Support)",
                    showlegend=True,
                    hoverinfo="text",
                    hovertext=[self._get_player_label(p, player_metadata) for p in players_by_zone[1]],
                )
            )

        # Plot Zone 0 player (ball carrier) - highlight with star
        if players_by_zone[0]:
            z0_player = players_by_zone[0][0]
            fig.add_trace(
                go.Scatter(
                    x=[z0_player["x"]],
                    y=[z0_player["y"]],
                    mode="markers",
                    marker=dict(
                        size=20,
                        color=self.colors["zone_0"],
                        line=dict(color="black", width=3),
                        symbol="star",
                    ),
                    name="Zone 0 (Ball Carrier)",
                    showlegend=True,
                    hoverinfo="text",
                    hovertext=[self._get_player_label(z0_player, player_metadata)],
                )
            )

            # Plot movement direction arrow if available
            if (
                z0_player.get("direction") is not None
                and z0_player.get("speed") is not None
            ):
                self._plot_movement_arrow(
                    fig,
                    z0_player["x"],
                    z0_player["y"],
                    z0_player["direction"],
                    z0_player["speed"],
                )

        # Add labels if enabled
        if show_labels:
            self._add_labels(fig, players, zone_assignments, player_metadata)

    def _plot_zone_boundaries(
        self,
        fig: go.Figure,
        zone_boundaries: Dict,
    ) -> None:
        """Plot zone boundary overlays."""
        zone_1_circle = zone_boundaries.get("zone_1_circle")
        zone_1_cone = zone_boundaries.get("zone_1_cone")

        if zone_1_circle:
            center = zone_1_circle["center"]
            radius = zone_1_circle["radius"]

            # Draw circle
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = center[0] + radius * np.cos(theta)
            circle_y = center[1] + radius * np.sin(theta)
            fig.add_trace(
                go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode="lines",
                    line=dict(color=self.colors["zone_1"], width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                    opacity=0.5,
                )
            )

        if zone_1_cone:
            center = zone_1_cone["center"]
            angle_start = zone_1_cone["angle_start"]
            angle_end = zone_1_cone["angle_end"]
            radius = zone_1_cone["radius"]

            # Draw cone (sector)
            angles = np.linspace(angle_start, angle_end, 50)
            x_points = [center[0] + radius * np.cos(angle) for angle in angles]
            y_points = [center[1] + radius * np.sin(angle) for angle in angles]

            # Create polygon for the cone (close the shape)
            cone_x = [center[0]] + x_points + [center[0]]
            cone_y = [center[1]] + y_points + [center[1]]

            fig.add_trace(
                go.Scatter(
                    x=cone_x,
                    y=cone_y,
                    mode="lines",
                    fill="toself",
                    fillcolor=self.colors["zone_1"],
                    line=dict(color=self.colors["zone_1"], width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                    opacity=0.1,
                )
            )

            # Draw lines for cone boundaries
            fig.add_trace(
                go.Scatter(
                    x=[center[0], center[0] + radius * np.cos(angle_start)],
                    y=[center[1], center[1] + radius * np.sin(angle_start)],
                    mode="lines",
                    line=dict(color=self.colors["zone_1"], width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                    opacity=0.5,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[center[0], center[0] + radius * np.cos(angle_end)],
                    y=[center[1], center[1] + radius * np.sin(angle_end)],
                    mode="lines",
                    line=dict(color=self.colors["zone_1"], width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                    opacity=0.5,
                )
            )

    def _add_labels(
        self,
        fig: go.Figure,
        players: List[Dict],
        zone_assignments: Dict[int, int],
        player_metadata: Optional[Dict] = None,
    ) -> None:
        """Add player labels to the plot (surnames)."""
        for player in players:
            player_id = player["player_id"]
            x, y = player["x"], player["y"]
            zone = zone_assignments.get(player_id, -1)

            # Get player surname (last name)
            label = self._get_player_label(player, player_metadata)
            if label is None:
                continue

            # Add text annotation
            fig.add_annotation(
                x=x,
                y=y,
                text=label,
                showarrow=False,
                font=dict(color="white", size=10, family="Arial Black"),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1,
            )

    def _get_player_label(
        self,
        player: Dict,
        player_metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """Get player label (surname or jersey number)."""
        player_id = player["player_id"]
        label = None
        meta = None

        if player_metadata:
            meta = player_metadata.get(player_id)
            # Try alternative type if not found
            if meta is None:
                if isinstance(player_id, int):
                    meta = player_metadata.get(str(player_id))
                elif isinstance(player_id, str):
                    try:
                        meta = player_metadata.get(int(player_id))
                    except (ValueError, TypeError):
                        pass

        if meta:
            if "name" in meta and meta["name"]:
                # Extract surname (last name)
                name_parts = meta["name"].strip().split()
                if len(name_parts) > 0:
                    label = name_parts[-1]  # Last name

            # If no surname found, try to use jersey number as fallback
            if label is None and "jersey_no" in meta and meta["jersey_no"]:
                label = str(meta["jersey_no"])

        return label

    def add_frame_info(
        self,
        fig: go.Figure,
        frame: int,
        timestamp: Optional[float] = None,
        period: Optional[int] = None,
        event_info: Optional[str] = None,
    ) -> None:
        """
        Add frame information text to the plot.

        Args:
            fig: Plotly figure
            frame: Frame number
            timestamp: Optional timestamp
            period: Optional period number
            event_info: Optional event information string
        """
        info_parts = [f"Frame: {frame}"]
        if period:
            info_parts.append(f"Period: {period}")
        if timestamp is not None:
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            info_parts.append(f"Time: {minutes:02d}:{seconds:02d}")
        if event_info:
            info_parts.append(event_info)

        info_text = " | ".join(info_parts)

        fig.add_annotation(
            x=-self.pitch_length / 2 + 2,
            y=self.pitch_width / 2 - 2,
            text=info_text,
            showarrow=False,
            xref="x",
            yref="y",
            xanchor="left",
            yanchor="top",
            font=dict(color="white", size=12),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1,
        )

    def _plot_trajectory(
        self,
        fig: go.Figure,
        trajectory_data: Dict,
    ) -> None:
        """
        Plot trajectory path for ball carrier.

        Args:
            fig: Plotly figure
            trajectory_data: Dictionary with 'x' and 'y' lists of positions
        """
        if "x" not in trajectory_data or "y" not in trajectory_data:
            return

        x_path = trajectory_data["x"]
        y_path = trajectory_data["y"]

        if len(x_path) < 2:
            return

        # Plot trajectory line with gradient effect
        # Create segments with varying opacity
        n_segments = len(x_path) - 1
        for i in range(n_segments):
            alpha = (i + 1) / n_segments * 0.7  # Fade from transparent to opaque
            fig.add_trace(
                go.Scatter(
                    x=[x_path[i], x_path[i + 1]],
                    y=[y_path[i], y_path[i + 1]],
                    mode="lines",
                    line=dict(
                        color=self.colors["zone_0"],
                        width=2.5,
                    ),
                    opacity=alpha,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Plot trajectory points (smaller, fading)
        for i, (x, y) in enumerate(zip(x_path[:-1], y_path[:-1])):
            alpha = (i + 1) / len(x_path) * 0.5
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=self.colors["zone_0"],
                        opacity=alpha,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    def _plot_movement_arrow(
        self,
        fig: go.Figure,
        x: float,
        y: float,
        direction: float,
        speed: float,
        arrow_length: float = 5.0,
    ) -> None:
        """
        Plot movement direction arrow for a player.

        Args:
            fig: Plotly figure
            x, y: Player position
            direction: Direction angle in radians
            speed: Speed in m/s
            arrow_length: Length of arrow in meters (scaled by speed)
        """
        # Scale arrow length by speed
        scaled_length = arrow_length * min(speed / 5.0, 2.0)

        # Calculate arrow end point
        dx = scaled_length * np.cos(direction)
        dy = scaled_length * np.sin(direction)

        # Plot arrow using annotation
        fig.add_annotation(
            x=x + dx,
            y=y + dy,
            ax=x,
            ay=y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=self.colors["zone_0"],
            arrowside="end",
        )

