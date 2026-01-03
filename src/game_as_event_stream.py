"""
Event Stream Game Representation

Represents a football game as a continuous stream of events with rich context.
Each event captures a change in game state with full contextual information.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class GameEvent:
    """A single event in the game stream"""
    event_id: str
    timestamp: float
    frame: int
    event_type: str
    player_id: Optional[int] = None
    team_id: Optional[int] = None
    context: Dict[str, Any] = None
    related_events: List[str] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.related_events is None:
            self.related_events = []


class EventStream:
    """Represents a game as a stream of events"""
    
    EVENT_TYPE_MAP = {
        "player_possession": ["possession_start", "possession_end"],
        "passing_option": ["option_available", "option_executed", "option_expired"],
        "off_ball_run": ["run_started", "run_ended"],
        "on_ball_engagement": ["pressing_started", "pressing_ended"]
    }
    
    def __init__(self, match_id: int):
        self.match_id = match_id
        self.events: List[GameEvent] = []
        self.event_index: Dict[str, GameEvent] = {}
        self.related_events_map: Dict[str, List[str]] = {}
        
    def add_event(self, event: GameEvent):
        """Add an event to the stream"""
        self.events.append(event)
        self.event_index[event.event_id] = event
        
        # Update related events mapping
        for related_id in event.related_events:
            if related_id not in self.related_events_map:
                self.related_events_map[related_id] = []
            self.related_events_map[related_id].append(event.event_id)
    
    def get_events_by_type(self, event_type: str) -> List[GameEvent]:
        """Get all events of a specific type"""
        return [e for e in self.events if e.event_type == event_type]
    
    def get_events_by_player(self, player_id: int) -> List[GameEvent]:
        """Get all events involving a specific player"""
        return [e for e in self.events if e.player_id == player_id]
    
    def get_events_by_time_range(self, start_time: float, end_time: float) -> List[GameEvent]:
        """Get events within a time range"""
        return [e for e in self.events if start_time <= e.timestamp <= end_time]
    
    def get_related_events(self, event_id: str) -> List[GameEvent]:
        """Get all events related to a given event"""
        related_ids = self.related_events_map.get(event_id, [])
        return [self.event_index[eid] for eid in related_ids if eid in self.event_index]
    
    def get_event_sequence(self, start_event_id: str, max_depth: int = 10) -> List[GameEvent]:
        """Get a sequence of related events starting from a given event"""
        sequence = []
        visited = set()
        queue = [(start_event_id, 0)]
        
        while queue:
            event_id, depth = queue.pop(0)
            if event_id in visited or depth > max_depth:
                continue
            
            visited.add(event_id)
            if event_id in self.event_index:
                event = self.event_index[event_id]
                sequence.append(event)
                
                # Add related events to queue
                for related_id in event.related_events:
                    if related_id not in visited:
                        queue.append((related_id, depth + 1))
        
        return sequence
    
    def find_patterns(self, pattern: List[str], max_gap: float = 10.0) -> List[List[GameEvent]]:
        """Find sequences of events matching a pattern"""
        matches = []
        
        for i, event in enumerate(self.events):
            if event.event_type == pattern[0]:
                sequence = [event]
                pattern_idx = 1
                
                for j in range(i + 1, len(self.events)):
                    next_event = self.events[j]
                    
                    # Check if time gap is too large
                    if next_event.timestamp - event.timestamp > max_gap:
                        break
                    
                    # Check if event matches pattern
                    if pattern_idx < len(pattern) and next_event.event_type == pattern[pattern_idx]:
                        sequence.append(next_event)
                        pattern_idx += 1
                        
                        # Found complete pattern
                        if pattern_idx == len(pattern):
                            matches.append(sequence)
                            break
        
        return matches
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert event stream to DataFrame"""
        events_data = []
        for event in self.events:
            event_dict = asdict(event)
            # Flatten context dict
            context = event_dict.pop("context", {})
            event_dict.update(context)
            events_data.append(event_dict)
        
        return pd.DataFrame(events_data)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "match_id": self.match_id,
            "n_events": len(self.events),
            "event_types": list(set(e.event_type for e in self.events)),
            "events": [asdict(e) for e in self.events]
        }


def build_event_stream_from_dynamic_events(events_df: pd.DataFrame, match_id: int) -> EventStream:
    """Build an event stream from SkillCorner dynamic events"""
    stream = EventStream(match_id)
    event_counter = 0
    
    for _, event_row in events_df.iterrows():
        timestamp = _time_to_seconds(event_row.get('time_start', '00:00.0'))
        frame_start = event_row.get('frame_start', 0)
        event_type_id = event_row.get('event_type_id')
        player_id = event_row.get('player_id')
        team_id = event_row.get('team_id')
        
        # Map event type
        if event_type_id == 8:  # player_possession
            # Create possession start event
            event_id = f"e_{event_counter:06d}"
            event_counter += 1
            
            possession_event = GameEvent(
                event_id=event_id,
                timestamp=timestamp,
                frame=int(frame_start),
                event_type="possession_start",
                player_id=int(player_id) if not pd.isna(player_id) else None,
                team_id=int(team_id) if not pd.isna(team_id) else None,
                context={
                    "location": {"x": event_row.get('x_start'), "y": event_row.get('y_start')},
                    "phase_type": event_row.get('team_in_possession_phase_type'),
                    "opponent_phase": event_row.get('team_out_of_possession_phase_type'),
                    "n_passing_options": event_row.get('n_passing_options', 0),
                    "n_off_ball_runs": event_row.get('n_off_ball_runs', 0),
                    "xthreat": event_row.get('xthreat'),
                    "xloss": event_row.get('xloss_player_possession_max'),
                    "game_state": event_row.get('game_state'),
                    "team_score": event_row.get('team_score', 0),
                    "opponent_score": event_row.get('opponent_team_score', 0)
                }
            )
            stream.add_event(possession_event)
            
            # Create possession end event
            end_timestamp = _time_to_seconds(event_row.get('time_end', '00:00.0'))
            end_event_id = f"e_{event_counter:06d}"
            event_counter += 1
            
            end_event = GameEvent(
                event_id=end_event_id,
                timestamp=end_timestamp,
                frame=int(event_row.get('frame_end', frame_start)),
                event_type="possession_end",
                player_id=int(player_id) if not pd.isna(player_id) else None,
                team_id=int(team_id) if not pd.isna(team_id) else None,
                context={
                    "end_type": event_row.get('end_type'),
                    "lead_to_shot": event_row.get('lead_to_shot', False),
                    "lead_to_goal": event_row.get('lead_to_goal', False),
                    "phase_transition": event_row.get('lead_to_different_phase', False)
                },
                related_events=[event_id]
            )
            stream.add_event(end_event)
        
        elif event_type_id == 7:  # passing_option
            event_id = f"e_{event_counter:06d}"
            event_counter += 1
            
            target_player = event_row.get('player_targeted_id')
            executed = event_row.get('received', False)
            
            option_event = GameEvent(
                event_id=event_id,
                timestamp=timestamp,
                frame=int(frame_start),
                event_type="option_available" if not executed else "option_executed",
                player_id=int(target_player) if not pd.isna(target_player) else None,
                team_id=int(team_id) if not pd.isna(team_id) else None,
                context={
                    "from_player": int(event_row.get('player_in_possession_id')) if not pd.isna(event_row.get('player_in_possession_id')) else None,
                    "location": {"x": event_row.get('x_start'), "y": event_row.get('y_start')},
                    "target_location": {"x": event_row.get('x_end'), "y": event_row.get('y_end')},
                    "xpass_completion": event_row.get('player_targeted_xpass_completion'),
                    "xthreat": event_row.get('player_targeted_xthreat'),
                    "dangerous": event_row.get('player_targeted_dangerous', False),
                    "difficult": event_row.get('player_targeted_difficult_pass_target', False),
                    "executed": executed
                },
                related_events=[f"e_{event_counter-2:06d}"] if event_counter > 1 else []  # Link to possession
            )
            stream.add_event(option_event)
        
        elif event_type_id == 9:  # on_ball_engagement
            event_id = f"e_{event_counter:06d}"
            event_counter += 1
            
            target_player = event_row.get('player_in_possession_id')
            subtype = event_row.get('event_subtype')
            
            engagement_event = GameEvent(
                event_id=event_id,
                timestamp=timestamp,
                frame=int(frame_start),
                event_type="pressing_started",
                player_id=int(player_id) if not pd.isna(player_id) else None,
                team_id=int(team_id) if not pd.isna(team_id) else None,
                context={
                    "target_player": int(target_player) if not pd.isna(target_player) else None,
                    "subtype": subtype,
                    "pressing_chain_id": event_row.get('pressing_chain'),
                    "pressing_chain_length": event_row.get('pressing_chain_length'),
                    "pressing_chain_index": event_row.get('index_in_pressing_chain'),
                    "beaten_by_possession": event_row.get('beaten_by_possession', False),
                    "beaten_by_movement": event_row.get('beaten_by_movement', False),
                    "distance": event_row.get('distance_covered')
                },
                related_events=[f"e_{event_counter-2:06d}"] if event_counter > 1 else []
            )
            stream.add_event(engagement_event)
        
        elif event_type_id == 1:  # off_ball_run
            event_id = f"e_{event_counter:06d}"
            event_counter += 1
            
            run_event = GameEvent(
                event_id=event_id,
                timestamp=timestamp,
                frame=int(frame_start),
                event_type="run_started",
                player_id=int(player_id) if not pd.isna(player_id) else None,
                team_id=int(team_id) if not pd.isna(team_id) else None,
                context={
                    "start_location": {"x": event_row.get('x_start'), "y": event_row.get('y_start')},
                    "end_location": {"x": event_row.get('x_end'), "y": event_row.get('y_end')},
                    "subtype": event_row.get('associated_off_ball_run_subtype'),
                    "distance_covered": event_row.get('distance_covered'),
                    "speed_avg": event_row.get('speed_avg'),
                    "trajectory_angle": event_row.get('trajectory_angle')
                },
                related_events=[f"e_{event_counter-2:06d}"] if event_counter > 1 else []
            )
            stream.add_event(run_event)
    
    return stream


def _time_to_seconds(time_str: str) -> float:
    """Convert time string (MM:SS.S) to seconds"""
    if pd.isna(time_str) or not time_str:
        return 0.0
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        return float(time_str)
    except:
        return 0.0


if __name__ == "__main__":
    # Example usage
    match_id = 1886347
    events_path = "../opendata/data/matches/1886347/1886347_dynamic_events.csv"
    events_df = pd.read_csv(events_path)
    
    # Build event stream
    stream = build_event_stream_from_dynamic_events(events_df, match_id)
    
    print(f"Event stream built: {len(stream.events)} events")
    print(f"\nEvent types: {set(e.event_type for e in stream.events)}")
    
    # Find patterns
    pattern = ["possession_start", "option_available", "option_executed"]
    matches = stream.find_patterns(pattern, max_gap=5.0)
    print(f"\nFound {len(matches)} matches for pattern: {pattern}")
    
    # Get events for a specific player
    player_events = stream.get_events_by_player(51649)
    print(f"\nEvents for player 51649: {len(player_events)}")
    
    # Convert to DataFrame
    df = stream.to_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

