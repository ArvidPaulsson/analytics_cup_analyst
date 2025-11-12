"""
Data loading module for zone visualizer.

Supports loading tracking data using kloppy or direct JSONL parsing.
Also loads dynamic events data for event integration.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

try:
    from kloppy import skillcorner

    KLOPPY_AVAILABLE = True
except ImportError:
    KLOPPY_AVAILABLE = False


class TrackingDataLoader:
    """Load and process tracking data from various sources."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the data loader.

        Args:
            data_dir: Base directory for data files. If None, assumes opendata structure.
        """
        self.data_dir = data_dir
        self.use_kloppy = KLOPPY_AVAILABLE

    def load_match_data(
        self,
        match_id: int,
        use_kloppy: Optional[bool] = None,
        coordinates: str = "skillcorner",
    ) -> Dict:
        """
        Load match tracking data and metadata.

        Args:
            match_id: Match ID to load
            use_kloppy: Whether to use kloppy (default: True if available)
            coordinates: Coordinate system to use (for kloppy)

        Returns:
            Dictionary containing:
            - tracking_data: DataFrame with frame-by-frame data
            - match_metadata: Match metadata (pitch dimensions, teams, etc.)
            - frames: List of frame dictionaries (if not using kloppy)
        """
        use_kloppy = use_kloppy if use_kloppy is not None else self.use_kloppy

        if use_kloppy and KLOPPY_AVAILABLE:
            return self._load_with_kloppy(match_id, coordinates)
        else:
            return self._load_from_jsonl(match_id)

    def _load_with_kloppy(self, match_id: int, coordinates: str) -> Dict:
        """Load tracking data using kloppy."""
        dataset = skillcorner.load_open_data(
            match_id=match_id,
            coordinates=coordinates,
        )

        # Convert to DataFrame
        df = dataset.to_df(engine="pandas")

        # Extract metadata
        home_team, away_team = dataset.metadata.teams
        pitch_length = dataset.metadata.pitch_dimensions.pitch_length
        pitch_width = dataset.metadata.pitch_dimensions.pitch_width

        # Get team directions from metadata
        # In kloppy, we can check the orientation
        match_metadata = {
            "match_id": match_id,
            "home_team": {
                "id": home_team.team_id,
                "name": home_team.name,
                "players": {
                    p.player_id: {
                        "name": p.name,
                        "jersey_no": p.jersey_no,
                        "starting_position": (
                            p.starting_position.name if p.starting_position else None
                        ),
                    }
                    for p in home_team.players
                },
            },
            "away_team": {
                "id": away_team.team_id,
                "name": away_team.name,
                "players": {
                    p.player_id: {
                        "name": p.name,
                        "jersey_no": p.jersey_no,
                        "starting_position": (
                            p.starting_position.name if p.starting_position else None
                        ),
                    }
                    for p in away_team.players
                },
            },
            "pitch_length": pitch_length,
            "pitch_width": pitch_width,
        }

        # Create player to team mapping
        # Handle both string and int player IDs (kloppy uses strings)
        player_to_team = {}
        for player in home_team.players:
            player_id = player.player_id
            # Store with original type (string from kloppy)
            player_to_team[player_id] = home_team.team_id
            # Also store as int for lookup flexibility
            try:
                player_to_team[int(player_id)] = home_team.team_id
            except (ValueError, TypeError):
                pass

        for player in away_team.players:
            player_id = player.player_id
            player_to_team[player_id] = away_team.team_id
            try:
                player_to_team[int(player_id)] = away_team.team_id
            except (ValueError, TypeError):
                pass

        # Transform kloppy DataFrame to our expected format
        df_normalized = self._normalize_kloppy_dataframe(
            df, home_team.team_id, away_team.team_id, player_to_team
        )

        return {
            "tracking_data": df_normalized,
            "match_metadata": match_metadata,
            "dataset": dataset,  # Keep dataset for metadata access
        }

    def _normalize_kloppy_dataframe(
        self,
        df: pd.DataFrame,
        home_team_id: int,
        away_team_id: int,
        player_to_team: Optional[Dict[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Normalize kloppy DataFrame to our expected format.

        Kloppy outputs wide format with columns like:
        - period_id, frame_id, timestamp
        - ball_x, ball_y, ball_owning_team_id
        - <player_id>_x, <player_id>_y

        We convert to long format with one row per player per frame.
        """
        rows = []

        # Get all player columns (format: <player_id>_x, <player_id>_y)
        player_x_cols = [
            col
            for col in df.columns
            if col.endswith("_x") and not col.startswith("ball_")
        ]
        # Extract player IDs from column names (e.g., "51009_x" -> 51009)
        player_ids = []
        for col in player_x_cols:
            try:
                player_id = int(col.replace("_x", ""))
                player_ids.append(player_id)
            except ValueError:
                continue

        for _, row in df.iterrows():
            frame = row.get("frame_id", 0)
            # Convert timestamp to seconds if it's a timedelta
            timestamp = row.get("timestamp", 0)
            if hasattr(timestamp, "total_seconds"):
                timestamp = timestamp.total_seconds()
            elif isinstance(timestamp, (int, float)):
                timestamp = float(timestamp)
            else:
                timestamp = 0

            period_id = row.get("period_id", 1)
            ball_x = row.get("ball_x", 0) if pd.notna(row.get("ball_x")) else 0
            ball_y = row.get("ball_y", 0) if pd.notna(row.get("ball_y")) else 0

            # Get ball owning team (possession)
            ball_owning_team_id = row.get("ball_owning_team_id")
            possession_group = None
            if pd.notna(ball_owning_team_id):
                if ball_owning_team_id == home_team_id:
                    possession_group = "home"
                elif ball_owning_team_id == away_team_id:
                    possession_group = "away"

            # Try to find ball carrier - check if any player is very close to ball
            ball_carrier_id = None
            if pd.notna(ball_x) and pd.notna(ball_y):
                min_dist = float("inf")
                for player_id in player_ids:
                    x_col = f"{player_id}_x"
                    y_col = f"{player_id}_y"
                    if x_col in df.columns and y_col in df.columns:
                        px = row.get(x_col)
                        py = row.get(y_col)
                        if pd.notna(px) and pd.notna(py):
                            # Calculate distance to ball
                            dist = ((px - ball_x) ** 2 + (py - ball_y) ** 2) ** 0.5
                            if dist < 2.0 and dist < min_dist:  # Within 2 meters
                                min_dist = dist
                                ball_carrier_id = player_id

            # Process each player
            for player_id in player_ids:
                x_col = f"{player_id}_x"
                y_col = f"{player_id}_y"
                d_col = f"{player_id}_d"  # Direction (angle)
                s_col = f"{player_id}_s"  # Speed

                if x_col in df.columns and pd.notna(row.get(x_col)):
                    x = row.get(x_col)
                    y = (
                        row.get(y_col)
                        if y_col in df.columns and pd.notna(row.get(y_col))
                        else 0
                    )

                    # Get velocity data if available
                    direction = (
                        row.get(d_col)
                        if d_col in df.columns and pd.notna(row.get(d_col))
                        else None
                    )
                    speed = (
                        row.get(s_col)
                        if s_col in df.columns and pd.notna(row.get(s_col))
                        else None
                    )

                    # Get team_id from player mapping
                    # Try both int and string versions
                    team_id = None
                    if player_to_team:
                        team_id = player_to_team.get(player_id)
                        if team_id is None:
                            # Try string version if player_id is int
                            if isinstance(player_id, int):
                                team_id = player_to_team.get(str(player_id))
                            # Try int version if player_id is string
                            elif isinstance(player_id, str):
                                try:
                                    team_id = player_to_team.get(int(player_id))
                                except ValueError:
                                    pass

                    rows.append(
                        {
                            "frame": frame,
                            "timestamp": timestamp,
                            "period_id": period_id,
                            "ball_x": ball_x,
                            "ball_y": ball_y,
                            "ball_carrier_id": ball_carrier_id,
                            "ball_owning_team_id": (
                                ball_owning_team_id
                                if pd.notna(ball_owning_team_id)
                                else None
                            ),
                            "possession_group": possession_group,
                            "player_id": player_id,
                            "x": x,
                            "y": y,
                            "team_id": team_id,
                            "direction": direction,
                            "speed": speed,
                        }
                    )

        return pd.DataFrame(rows)

    def _load_from_jsonl(self, match_id: int) -> Dict:
        """Load tracking data directly from JSONL files."""
        if self.data_dir is None:
            # Assume opendata structure
            base_path = (
                Path(__file__).parent.parent.parent.parent
                / "opendata"
                / "data"
                / "matches"
            )
        else:
            base_path = Path(self.data_dir) / "matches"

        match_path = base_path / str(match_id)
        tracking_file = match_path / f"{match_id}_tracking_extrapolated.jsonl"
        match_file = match_path / f"{match_id}_match.json"

        if not tracking_file.exists():
            raise FileNotFoundError(f"Tracking file not found: {tracking_file}")

        # Load tracking data
        frames = []
        with open(tracking_file, "r") as f:
            for line in f:
                if line.strip():
                    frames.append(json.loads(line))

        # Load match metadata
        match_metadata = {}
        if match_file.exists():
            with open(match_file, "r") as f:
                match_metadata = json.load(f)

        # Convert to DataFrame-like structure
        tracking_data = self._frames_to_dataframe(frames)

        return {
            "tracking_data": tracking_data,
            "match_metadata": match_metadata,
            "frames": frames,
        }

    def _frames_to_dataframe(self, frames: List[Dict]) -> pd.DataFrame:
        """Convert frame list to DataFrame format compatible with kloppy output."""
        rows = []
        for frame in frames:
            frame_num = frame.get("frame")
            timestamp = frame.get("timestamp", 0)
            period = frame.get("period", 1)

            # Ball data
            ball_data = frame.get("ball_data", {})
            ball_x = ball_data.get("x", 0)
            ball_y = ball_data.get("y", 0)

            # Possession
            possession = frame.get("possession", {})
            ball_carrier_id = possession.get("player_id")
            possession_group = possession.get("group")  # 'home' or 'away'

            # Player data
            player_data = frame.get("player_data", [])
            for player in player_data:
                player_id = player.get("player_id")
                x = player.get("x", 0)
                y = player.get("y", 0)

                # Determine team (this is simplified - may need match metadata)
                # In possession data, we can infer team from group
                team_id = None  # Will need to map from match metadata

                rows.append(
                    {
                        "frame": frame_num,
                        "timestamp": timestamp,
                        "period_id": period,
                        "ball_x": ball_x,
                        "ball_y": ball_y,
                        "ball_carrier_id": ball_carrier_id,
                        "possession_group": possession_group,
                        "player_id": player_id,
                        "x": x,
                        "y": y,
                        "team_id": team_id,
                    }
                )

        return pd.DataFrame(rows)

    def load_dynamic_events(self, match_id: int) -> pd.DataFrame:
        """
        Load dynamic events data.

        Args:
            match_id: Match ID

        Returns:
            DataFrame with dynamic events
        """
        if self.data_dir is None:
            base_path = (
                Path(__file__).parent.parent.parent.parent
                / "opendata"
                / "data"
                / "matches"
            )
        else:
            base_path = Path(self.data_dir) / "matches"

        match_path = base_path / str(match_id)
        events_file = match_path / f"{match_id}_dynamic_events.csv"

        if not events_file.exists():
            raise FileNotFoundError(f"Events file not found: {events_file}")

        events_df = pd.read_csv(events_file)
        return events_df

    def get_frame_data(
        self,
        tracking_data: pd.DataFrame,
        frame: int,
        match_metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Extract data for a specific frame.

        Args:
            tracking_data: Full tracking DataFrame
            frame: Frame number
            match_metadata: Optional match metadata

        Returns:
            Dictionary with frame data:
            - ball_position: (x, y)
            - ball_carrier_id: Player ID with ball
            - players: List of player dicts with (x, y, player_id, team_id)
        """
        frame_data = tracking_data[tracking_data["frame"] == frame].copy()

        if frame_data.empty:
            return {
                "ball_position": (0, 0),
                "ball_carrier_id": None,
                "players": [],
            }

        # Get ball position (should be same for all rows in frame)
        ball_x = (
            frame_data["ball_x"].iloc[0]
            if pd.notna(frame_data["ball_x"].iloc[0])
            else 0
        )
        ball_y = (
            frame_data["ball_y"].iloc[0]
            if pd.notna(frame_data["ball_y"].iloc[0])
            else 0
        )
        ball_carrier_id = (
            frame_data["ball_carrier_id"].iloc[0]
            if pd.notna(frame_data["ball_carrier_id"].iloc[0])
            else None
        )

        # Get ball owning team ID (possession team) - more reliable than ball_carrier_id
        ball_owning_team_id = None
        if "ball_owning_team_id" in frame_data.columns:
            ball_owning_team_id = (
                frame_data["ball_owning_team_id"].iloc[0]
                if pd.notna(frame_data["ball_owning_team_id"].iloc[0])
                else None
            )

        # Get all players
        players = []
        for _, row in frame_data.iterrows():
            if pd.notna(row.get("x")) and pd.notna(row.get("y")):
                player_id = row["player_id"]
                team_id = row.get("team_id") if pd.notna(row.get("team_id")) else None

                # If team_id is None, try to infer from match metadata
                if team_id is None and match_metadata:
                    # Check if player is in home or away team
                    for team_key in ["home_team", "away_team"]:
                        if team_key in match_metadata:
                            team = match_metadata[team_key]
                            if "players" in team and player_id in team["players"]:
                                team_id = team.get("id")
                                break

                players.append(
                    {
                        "player_id": player_id,
                        "x": row["x"],
                        "y": row["y"],
                        "team_id": team_id,
                        "direction": (
                            row.get("direction")
                            if pd.notna(row.get("direction"))
                            else None
                        ),
                        "speed": (
                            row.get("speed") if pd.notna(row.get("speed")) else None
                        ),
                    }
                )

        return {
            "ball_position": (ball_x, ball_y),
            "ball_carrier_id": ball_carrier_id,
            "ball_owning_team_id": ball_owning_team_id,
            "players": players,
        }

    def get_ball_carrier_movement(
        self,
        tracking_data: pd.DataFrame,
        frame: int,
        lookback_frames: int = 5,
    ) -> Optional[Dict]:
        """
        Calculate ball carrier movement direction and speed.

        Args:
            tracking_data: Full tracking DataFrame
            frame: Current frame number
            lookback_frames: Number of frames to look back for direction calculation

        Returns:
            Dictionary with:
            - direction: Angle in radians (0 = right, π/2 = up)
            - speed: Speed in m/s
            - velocity_x: Velocity in x direction (m/s)
            - velocity_y: Velocity in y direction (m/s)
            - or None if ball carrier not found or insufficient data
        """
        import numpy as np

        current_frame_data = tracking_data[tracking_data["frame"] == frame]
        if current_frame_data.empty:
            return None

        ball_carrier_id = current_frame_data["ball_carrier_id"].iloc[0]
        if pd.isna(ball_carrier_id):
            return None

        # Try to get velocity from kloppy data first
        carrier_data = current_frame_data[
            current_frame_data["player_id"] == ball_carrier_id
        ]
        if not carrier_data.empty:
            direction = carrier_data["direction"].iloc[0]
            speed = carrier_data["speed"].iloc[0]

            if pd.notna(direction) and pd.notna(speed):
                # Convert direction (likely in degrees) to radians
                direction_rad = (
                    np.deg2rad(direction) if abs(direction) > 2 * np.pi else direction
                )
                velocity_x = speed * np.cos(direction_rad)
                velocity_y = speed * np.sin(direction_rad)

                return {
                    "direction": direction_rad,
                    "speed": speed,
                    "velocity_x": velocity_x,
                    "velocity_y": velocity_y,
                }

        # Fallback: calculate from position differences
        prev_frame = frame - lookback_frames
        if prev_frame < tracking_data["frame"].min():
            return None

        prev_frame_data = tracking_data[tracking_data["frame"] == prev_frame]
        if prev_frame_data.empty:
            return None

        prev_carrier = prev_frame_data[prev_frame_data["player_id"] == ball_carrier_id]
        if prev_carrier.empty:
            return None

        # Calculate velocity from position change
        dx = carrier_data["x"].iloc[0] - prev_carrier["x"].iloc[0]
        dy = carrier_data["y"].iloc[0] - prev_carrier["y"].iloc[0]
        dt = carrier_data["timestamp"].iloc[0] - prev_carrier["timestamp"].iloc[0]

        if dt <= 0:
            return None

        velocity_x = dx / dt
        velocity_y = dy / dt
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        direction = np.arctan2(dy, dx)

        return {
            "direction": direction,
            "speed": speed,
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
        }

    def get_attacking_direction(
        self,
        match_metadata: Dict,
        period: int,
        team_id: Optional[int] = None,
    ) -> str:
        """
        Determine attacking direction for a team in a period.

        Args:
            match_metadata: Match metadata
            period: Period number (1 or 2)
            team_id: Team ID (if None, returns for home team)

        Returns:
            'left_to_right' or 'right_to_left'
        """
        # Try to get from metadata
        if "home_team_side" in match_metadata:
            sides = match_metadata["home_team_side"]
            if period == 1:
                return sides[0] if len(sides) > 0 else "left_to_right"
            else:
                return (
                    sides[1]
                    if len(sides) > 1
                    else sides[0] if len(sides) > 0 else "left_to_right"
                )

        # Default: home team attacks left to right in period 1
        if period == 1:
            return "left_to_right"
        else:
            return "right_to_left"
