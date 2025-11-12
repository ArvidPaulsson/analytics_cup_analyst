"""
Zone calculation module for assigning players to zones.

Implements Seirul·lo's three-zone system:
- Zone 0: Ball carrier (individual skills)
- Zone 1: Mutual support zone (movement and support)
- Zone 2: Cooperation zone (tactical positioning)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class ZoneConfig:
    """Configuration for zone calculations."""

    zone_0_radius: float = 5.0  # meters - Zone 0 is just the ball carrier
    zone_1_radius: float = 15.0  # meters - Support zone radius
    zone_1_angle: float = 60.0  # degrees - Angular cone for Zone 1 (±angle)
    zone_1_orientation: str = "toward_goal"  # 'toward_goal' or 'custom'
    zone_1_custom_angle: Optional[float] = None  # Custom angle in degrees (0-360)
    zone_2_radius: Optional[float] = None  # None means all remaining players

    def __post_init__(self):
        """Validate configuration."""
        if self.zone_1_orientation == "custom" and self.zone_1_custom_angle is None:
            raise ValueError(
                "zone_1_custom_angle must be set when zone_1_orientation is 'custom'"
            )


class ZoneCalculator:
    """Calculate zone assignments for players."""

    def __init__(self, config: Optional[ZoneConfig] = None):
        """
        Initialize zone calculator.

        Args:
            config: Zone configuration. If None, uses default.
        """
        self.config = config or ZoneConfig()

    def calculate_zones(
        self,
        ball_position: Tuple[float, float],
        ball_carrier_id: Optional[int],
        players: List[Dict],
        ball_carrier_position: Optional[Tuple[float, float]] = None,
        attacking_direction: str = "left_to_right",
        goal_position: Optional[Tuple[float, float]] = None,
        ball_movement_direction: Optional[float] = None,
    ) -> Dict[int, int]:
        """
        Calculate zone assignments for all players.

        Args:
            ball_position: (x, y) position of the ball
            ball_carrier_id: Player ID of ball carrier (None if no carrier)
            players: List of player dicts with keys: player_id, x, y, team_id, direction, speed
            ball_carrier_position: Optional explicit position of ball carrier
            attacking_direction: 'left_to_right' or 'right_to_left'
            goal_position: Optional explicit goal position (x, y)
            ball_movement_direction: Optional movement direction in radians (0 = right, π/2 = up)

        Returns:
            Dictionary mapping player_id -> zone (0, 1, 2, or -1 for opponents/not assigned)
        """
        if ball_carrier_id is None:
            # No ball carrier - assign all players to Zone 2 or unassigned
            return {p["player_id"]: 2 for p in players}

        # Find ball carrier position and movement direction
        ball_carrier_pos = ball_carrier_position
        ball_carrier_movement_dir = ball_movement_direction

        # Extract ball carrier position if not provided
        if ball_carrier_pos is None:
            for player in players:
                if player["player_id"] == ball_carrier_id:
                    ball_carrier_pos = (player["x"], player["y"])
                    break

        # Extract movement direction if not provided
        if ball_carrier_movement_dir is None:
            for player in players:
                if player["player_id"] == ball_carrier_id:
                    # Try to get movement direction from player data
                    if player.get("direction") is not None:
                        direction = player["direction"]
                        # Convert from degrees to radians if needed
                        if abs(direction) > 2 * np.pi:
                            ball_carrier_movement_dir = np.deg2rad(direction)
                        else:
                            ball_carrier_movement_dir = direction
                    break

        if ball_carrier_pos is None:
            # Ball carrier not found in players list
            return {p["player_id"]: 2 for p in players}

        # Calculate Zone 1 orientation
        # Priority: ball movement direction > custom angle > goal direction
        if ball_carrier_movement_dir is not None:
            # Use ball movement direction
            zone_1_dir = ball_carrier_movement_dir
        elif self.config.zone_1_orientation == "toward_goal":
            zone_1_dir = self._get_goal_direction(
                ball_carrier_pos,
                attacking_direction,
                goal_position,
            )
        else:
            # Custom angle
            zone_1_dir = np.deg2rad(self.config.zone_1_custom_angle)

        zone_assignments = {}

        for player in players:
            player_id = player["player_id"]
            player_pos = (player["x"], player["y"])

            # Zone 0: Ball carrier only
            if player_id == ball_carrier_id:
                zone_assignments[player_id] = 0
                continue

            # Calculate distance from ball carrier
            distance = self._euclidean_distance(ball_carrier_pos, player_pos)

            # Zone 1: Within radius AND within angle cone
            if distance <= self.config.zone_1_radius:
                # Check if within angle cone
                angle_to_player = self._angle_between_points(
                    ball_carrier_pos,
                    player_pos,
                )
                angle_diff = self._angle_difference(zone_1_dir, angle_to_player)

                if abs(angle_diff) <= np.deg2rad(self.config.zone_1_angle):
                    zone_assignments[player_id] = 1
                    continue

            # Zone 2: Beyond Zone 1 radius (or all remaining if zone_2_radius is None)
            if (
                self.config.zone_2_radius is None
                or distance > self.config.zone_1_radius
            ):
                zone_assignments[player_id] = 2
            else:
                # Between Zone 1 and Zone 2 radius - assign to Zone 2
                zone_assignments[player_id] = 2

        return zone_assignments

    def _get_goal_direction(
        self,
        position: Tuple[float, float],
        attacking_direction: str,
        goal_position: Optional[Tuple[float, float]] = None,
    ) -> float:
        """
        Calculate angle toward attacking goal.

        Args:
            position: Current position (x, y)
            attacking_direction: 'left_to_right' or 'right_to_left'
            goal_position: Optional explicit goal position

        Returns:
            Angle in radians toward goal
        """
        if goal_position is not None:
            return self._angle_between_points(position, goal_position)

        # Default: assume goal is at x = ±52.5 (half pitch length) based on direction
        x, y = position
        if attacking_direction == "left_to_right":
            # Attacking right, goal is at high x
            goal_x = 52.5
        else:
            # Attacking left, goal is at low x
            goal_x = -52.5

        goal_y = y  # Same y-coordinate
        return self._angle_between_points(position, (goal_x, goal_y))

    def _euclidean_distance(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float],
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _angle_between_points(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float],
    ) -> float:
        """
        Calculate angle from pos1 to pos2.

        Returns:
            Angle in radians (0 = right, π/2 = up, -π/2 = down)
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return np.arctan2(dy, dx)

    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """
        Calculate smallest difference between two angles.

        Returns:
            Difference in radians, normalized to [-π, π]
        """
        diff = angle2 - angle1
        # Normalize to [-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    def get_zone_boundaries(
        self,
        ball_carrier_position: Tuple[float, float],
        attacking_direction: str = "left_to_right",
        goal_position: Optional[Tuple[float, float]] = None,
        ball_movement_direction: Optional[float] = None,
    ) -> Dict:
        """
        Get zone boundary geometries for visualization.

        Args:
            ball_carrier_position: Position of ball carrier
            attacking_direction: 'left_to_right' or 'right_to_left'
            goal_position: Optional explicit goal position
            ball_movement_direction: Optional movement direction in radians

        Returns:
            Dictionary with zone boundary data:
            - zone_1_circle: (center, radius)
            - zone_1_cone: (center, angle_start, angle_end, radius)
        """
        # Zone 1 circle
        zone_1_circle = {
            "center": ball_carrier_position,
            "radius": self.config.zone_1_radius,
        }

        # Zone 1 cone orientation
        # Priority: ball movement direction > custom angle > goal direction
        if ball_movement_direction is not None:
            zone_1_dir = ball_movement_direction
        elif self.config.zone_1_orientation == "toward_goal":
            zone_1_dir = self._get_goal_direction(
                ball_carrier_position,
                attacking_direction,
                goal_position,
            )
        else:
            zone_1_dir = np.deg2rad(self.config.zone_1_custom_angle)

        angle_start = zone_1_dir - np.deg2rad(self.config.zone_1_angle)
        angle_end = zone_1_dir + np.deg2rad(self.config.zone_1_angle)

        zone_1_cone = {
            "center": ball_carrier_position,
            "angle_start": angle_start,
            "angle_end": angle_end,
            "radius": self.config.zone_1_radius,
        }

        return {
            "zone_1_circle": zone_1_circle,
            "zone_1_cone": zone_1_cone,
        }
