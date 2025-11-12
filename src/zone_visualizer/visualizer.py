"""
Visualization module for zone overlays on mplsoccer Pitch.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class ZoneVisualizer:
    """Visualize zones on mplsoccer Pitch."""

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        pitch_type: str = "skillcorner",
    ):
        """
        Initialize visualizer.

        Args:
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
            pitch_type: Pitch type for mplsoccer
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.pitch_type = pitch_type

        # Color scheme
        self.colors = {
            "zone_0": "#FFD700",  # Gold
            "zone_1": "#00CED1",  # Dark turquoise
            "zone_2": "#4169E1",  # Royal blue
            "opponent": "#DC143C",  # Crimson
            "ball": "#FFFFFF",  # White
        }

        # Zone labels
        self.zone_labels = {
            0: "Zone 0",
            1: "Zone 1",
            2: "Zone 2",
        }

    def create_pitch(
        self, ax: Optional[plt.Axes] = None
    ) -> Tuple[Pitch, plt.Figure, plt.Axes]:
        """
        Create mplsoccer Pitch.

        Args:
            ax: Optional existing axes

        Returns:
            Tuple of (pitch, figure, axes)
        """
        pitch = Pitch(
            pitch_type=self.pitch_type,
            line_alpha=0.75,
            pitch_length=self.pitch_length,
            pitch_width=self.pitch_width,
            pitch_color="#001400",
            line_color="white",
            linewidth=1.5,
        )

        if ax is None:
            fig, ax = pitch.grid(figheight=10, endnote_height=0, title_height=0)
        else:
            fig = ax.figure

        return pitch, fig, ax

    def plot_frame(
        self,
        ax: plt.Axes,
        pitch: Pitch,
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
        Plot a single frame with zone overlays.

        Args:
            ax: Matplotlib axes
            pitch: mplsoccer Pitch object
            frame_data: Frame data dict with ball_position, ball_carrier_id, players
            zone_assignments: Dict mapping player_id -> zone
            show_overlays: Whether to show zone overlays
            show_labels: Whether to show zone labels
            show_zone_boundaries: Whether to show zone boundaries
            zone_boundaries: Optional zone boundary data from ZoneCalculator
            player_metadata: Optional dict mapping player_id -> {name, jersey_no, team_id}
        """
        # Clear previous plots (but keep pitch)
        # We'll plot on top of the pitch

        ball_position = frame_data["ball_position"]
        ball_carrier_id = frame_data["ball_carrier_id"]
        players = frame_data["players"]

        # Get possession team from frame data
        # First try ball_owning_team_id (most reliable)
        possession_team_id = frame_data.get("ball_owning_team_id")

        # If not available, try to get from ball carrier
        if possession_team_id is None and ball_carrier_id:
            for player in players:
                if player["player_id"] == ball_carrier_id:
                    possession_team_id = player.get("team_id")
                    break

        # If we still don't have team, try to infer from zone assignments
        # (players in zones 0, 1, 2 are teammates)
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
            self._plot_trajectory(ax, pitch, trajectory_data)

        # Plot ball
        pitch.scatter(
            ball_position[0],
            ball_position[1],
            ax=ax,
            s=200,
            color=self.colors["ball"],
            edgecolors="black",
            linewidths=2,
            zorder=20,
            marker="o",
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
            # Use possession_team_id if available, otherwise use zone assignment
            if possession_team_id is not None and player_team_id is not None:
                is_teammate = player_team_id == possession_team_id
            else:
                # Fallback: if player is in a zone (0, 1, 2), they're a teammate
                is_teammate = zone in [0, 1, 2]

            # Assign to zone if teammate and in a valid zone
            if is_teammate and zone in [0, 1, 2]:
                players_by_zone[zone].append(player)
            else:
                # Opponent or unassigned (not in possession team's zones)
                opponents.append(player)

        # Plot zone boundaries if enabled
        if show_overlays and show_zone_boundaries and zone_boundaries:
            self._plot_zone_boundaries(ax, pitch, zone_boundaries)

        # Plot opponents (clearly distinguished with square markers)
        if opponents:
            opp_x = [p["x"] for p in opponents]
            opp_y = [p["y"] for p in opponents]
            pitch.scatter(
                opp_x,
                opp_y,
                ax=ax,
                s=300,
                color=self.colors["opponent"],
                edgecolors="white",
                linewidths=2,
                zorder=10,
                alpha=0.9,
                marker="s",  # Square markers for opponents
            )

        # Plot Zone 2 players (cooperation zone)
        if players_by_zone[2]:
            z2_x = [p["x"] for p in players_by_zone[2]]
            z2_y = [p["y"] for p in players_by_zone[2]]
            pitch.scatter(
                z2_x,
                z2_y,
                ax=ax,
                s=300,
                color=self.colors["zone_2"],
                edgecolors="white",
                linewidths=1.5,
                zorder=11,
                alpha=0.8,
            )

        # Plot Zone 1 players (support zone)
        if players_by_zone[1]:
            z1_x = [p["x"] for p in players_by_zone[1]]
            z1_y = [p["y"] for p in players_by_zone[1]]
            pitch.scatter(
                z1_x,
                z1_y,
                ax=ax,
                s=350,
                color=self.colors["zone_1"],
                edgecolors="white",
                linewidths=2,
                zorder=12,
                alpha=0.9,
            )

        # Plot Zone 0 player (ball carrier) - highlight
        if players_by_zone[0]:
            z0_player = players_by_zone[0][0]
            pitch.scatter(
                z0_player["x"],
                z0_player["y"],
                ax=ax,
                s=400,
                color=self.colors["zone_0"],
                edgecolors="black",
                linewidths=3,
                zorder=15,
                marker="*",
            )

            # Plot movement direction arrow if available
            if (
                z0_player.get("direction") is not None
                and z0_player.get("speed") is not None
            ):
                self._plot_movement_arrow(
                    ax,
                    pitch,
                    z0_player["x"],
                    z0_player["y"],
                    z0_player["direction"],
                    z0_player["speed"],
                )

        # Add labels if enabled
        if show_labels:
            self._add_labels(
                ax,
                pitch,
                players,
                zone_assignments,
                player_metadata,
            )

    def _plot_zone_boundaries(
        self,
        ax: plt.Axes,
        pitch: Pitch,
        zone_boundaries: Dict,
    ) -> None:
        """Plot zone boundary overlays."""
        zone_1_circle = zone_boundaries.get("zone_1_circle")
        zone_1_cone = zone_boundaries.get("zone_1_cone")

        if zone_1_circle:
            center = zone_1_circle["center"]
            radius = zone_1_circle["radius"]

            # Draw circle
            circle = plt.Circle(
                center,
                radius,
                fill=False,
                edgecolor=self.colors["zone_1"],
                linewidth=2,
                linestyle="--",
                alpha=0.5,
                zorder=5,
            )
            ax.add_patch(circle)

        if zone_1_cone:
            center = zone_1_cone["center"]
            angle_start = zone_1_cone["angle_start"]
            angle_end = zone_1_cone["angle_end"]
            radius = zone_1_cone["radius"]

            # Draw cone (sector)
            # Create points for the arc
            angles = np.linspace(angle_start, angle_end, 50)
            x_points = [center[0] + radius * np.cos(angle) for angle in angles]
            y_points = [center[1] + radius * np.sin(angle) for angle in angles]

            # Create polygon for the cone
            cone_points = [(center[0], center[1])] + list(zip(x_points, y_points))
            cone_polygon = mpatches.Polygon(
                cone_points,
                fill=True,
                edgecolor=self.colors["zone_1"],
                facecolor=self.colors["zone_1"],
                alpha=0.1,
                linewidth=2,
                linestyle="--",
                zorder=4,
            )
            ax.add_patch(cone_polygon)

            # Draw lines for cone boundaries
            ax.plot(
                [center[0], center[0] + radius * np.cos(angle_start)],
                [center[1], center[1] + radius * np.sin(angle_start)],
                color=self.colors["zone_1"],
                linewidth=2,
                linestyle="--",
                alpha=0.5,
                zorder=5,
            )
            ax.plot(
                [center[0], center[0] + radius * np.cos(angle_end)],
                [center[1], center[1] + radius * np.sin(angle_end)],
                color=self.colors["zone_1"],
                linewidth=2,
                linestyle="--",
                alpha=0.5,
                zorder=5,
            )

    def _add_labels(
        self,
        ax: plt.Axes,
        pitch: Pitch,
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
            # Try both int and string player_id for lookup
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

            # If still no label, skip (don't show player ID)
            if label is None:
                continue

            # Add text label
            pitch.annotate(
                label,
                xy=(x, y),
                ax=ax,
                fontsize=9,
                ha="center",
                va="center",
                color="white",
                weight="bold",
                zorder=20,
            )

    def add_frame_info(
        self,
        ax: plt.Axes,
        frame: int,
        timestamp: Optional[float] = None,
        period: Optional[int] = None,
        event_info: Optional[str] = None,
    ) -> None:
        """
        Add frame information text to the plot.

        Args:
            ax: Matplotlib axes
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
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            color="white",
            zorder=30,
        )

    def _plot_trajectory(
        self,
        ax: plt.Axes,
        pitch: Pitch,
        trajectory_data: Dict,
    ) -> None:
        """
        Plot trajectory path for ball carrier.

        Args:
            ax: Matplotlib axes
            pitch: mplsoccer Pitch object
            trajectory_data: Dictionary with 'x' and 'y' lists of positions
        """
        if "x" not in trajectory_data or "y" not in trajectory_data:
            return

        x_path = trajectory_data["x"]
        y_path = trajectory_data["y"]

        if len(x_path) < 2:
            return

        # Plot trajectory line with gradient (fade from old to new)
        for i in range(len(x_path) - 1):
            alpha = (i + 1) / len(x_path)  # Fade from transparent to opaque
            pitch.plot(
                [x_path[i], x_path[i + 1]],
                [y_path[i], y_path[i + 1]],
                ax=ax,
                color=self.colors["zone_0"],
                linewidth=2.5,
                alpha=alpha * 0.7,
                zorder=3,
            )

        # Plot trajectory points (smaller, fading)
        for i, (x, y) in enumerate(zip(x_path[:-1], y_path[:-1])):  # All but last point
            alpha = (i + 1) / len(x_path) * 0.5
            pitch.scatter(
                x,
                y,
                ax=ax,
                s=50,
                color=self.colors["zone_0"],
                alpha=alpha,
                zorder=4,
            )

    def _plot_movement_arrow(
        self,
        ax: plt.Axes,
        pitch: Pitch,
        x: float,
        y: float,
        direction: float,
        speed: float,
        arrow_length: float = 5.0,
    ) -> None:
        """
        Plot movement direction arrow for a player.

        Args:
            ax: Matplotlib axes
            pitch: mplsoccer Pitch object
            x, y: Player position
            direction: Direction angle in radians
            speed: Speed in m/s
            arrow_length: Length of arrow in meters (scaled by speed)
        """
        # Scale arrow length by speed (normalize to 0-10 m/s range)
        scaled_length = arrow_length * min(speed / 5.0, 2.0)  # Max 2x length

        # Calculate arrow end point
        dx = scaled_length * np.cos(direction)
        dy = scaled_length * np.sin(direction)

        # Plot arrow
        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=1.5,
            head_length=1.0,
            fc=self.colors["zone_0"],
            ec="black",
            linewidth=2,
            zorder=14,
            alpha=0.8,
        )

    def add_legend(self, ax: plt.Axes) -> None:
        """Add legend to the plot."""
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor=self.colors["zone_0"],
                markersize=12,
                label="Zone 0 (Ball Carrier)",
                markeredgecolor="black",
                markeredgewidth=2,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.colors["zone_1"],
                markersize=10,
                label="Zone 1 (Support)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.colors["zone_2"],
                markersize=10,
                label="Zone 2 (Cooperation)",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=self.colors["opponent"],
                markersize=10,
                label="Opponents",
            ),
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=9,
            framealpha=0.9,
        )
