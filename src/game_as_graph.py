"""
Graph-Based Game Representation

Represents a football game as a dynamic graph where:
- Nodes = Players, Ball, Zones, Phases
- Edges = Passes, Pressures, Runs, Spatial Relationships
- Attributes = Time, Location, Context
"""

import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class GraphNode:
    """A node in the game graph"""
    node_id: str
    node_type: str  # 'player', 'ball', 'zone', 'phase'
    attributes: Dict


@dataclass
class GraphEdge:
    """An edge in the game graph"""
    source: str
    target: str
    edge_type: str  # 'pass', 'pressure', 'run', 'spatial', 'option'
    timestamp: float
    attributes: Dict


class GameGraph:
    """Represents a football game as a dynamic graph"""
    
    def __init__(self, match_id: int):
        self.match_id = match_id
        self.graph = nx.MultiDiGraph()  # MultiDiGraph allows multiple edges between nodes
        self.nodes_by_type = defaultdict(list)
        self.temporal_snapshots = []
        
    def add_player_node(self, player_id: int, team_id: int, timestamp: float, 
                       position: Tuple[float, float], attributes: Dict = None):
        """Add a player node to the graph"""
        node_id = f"player_{player_id}"
        node_data = {
            "node_type": "player",
            "team_id": team_id,
            "timestamp": timestamp,
            "position": position,
            **(attributes or {})
        }
        
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, **node_data)
            self.nodes_by_type["player"].append(node_id)
        else:
            # Update existing node
            self.graph.nodes[node_id].update(node_data)
    
    def add_pass_edge(self, from_player: int, to_player: int, timestamp: float,
                     successful: bool, attributes: Dict = None):
        """Add a pass edge between players"""
        source = f"player_{from_player}"
        target = f"player_{to_player}"
        
        edge_data = {
            "edge_type": "pass",
            "timestamp": timestamp,
            "successful": successful,
            **(attributes or {})
        }
        
        self.graph.add_edge(source, target, **edge_data)
    
    def add_passing_option_edge(self, from_player: int, to_player: int, 
                               timestamp: float, executed: bool, attributes: Dict = None):
        """Add a passing option edge (available but may not be executed)"""
        source = f"player_{from_player}"
        target = f"player_{to_player}"
        
        edge_data = {
            "edge_type": "passing_option",
            "timestamp": timestamp,
            "executed": executed,
            **(attributes or {})
        }
        
        self.graph.add_edge(source, target, **edge_data)
    
    def add_pressure_edge(self, presser: int, target: int, timestamp: float,
                         attributes: Dict = None):
        """Add a pressure/pressing edge"""
        source = f"player_{presser}"
        target = f"player_{target}"
        
        edge_data = {
            "edge_type": "pressure",
            "timestamp": timestamp,
            **(attributes or {})
        }
        
        self.graph.add_edge(source, target, **edge_data)
    
    def add_off_ball_run_edge(self, runner: int, target_location: Tuple[float, float],
                             timestamp: float, attributes: Dict = None):
        """Add an off-ball run edge (spatial movement)"""
        source = f"player_{runner}"
        target = f"zone_{int(target_location[0])}_{int(target_location[1])}"
        
        # Ensure zone node exists
        if not self.graph.has_node(target):
            self.graph.add_node(target, node_type="zone", position=target_location)
            self.nodes_by_type["zone"].append(target)
        
        edge_data = {
            "edge_type": "off_ball_run",
            "timestamp": timestamp,
            **(attributes or {})
        }
        
        self.graph.add_edge(source, target, **edge_data)
    
    def snapshot(self, timestamp: float):
        """Create a temporal snapshot of the graph"""
        snapshot = {
            "timestamp": timestamp,
            "nodes": dict(self.graph.nodes(data=True)),
            "edges": list(self.graph.edges(data=True))
        }
        self.temporal_snapshots.append(snapshot)
    
    def get_passing_network(self, team_id: Optional[int] = None, 
                           successful_only: bool = True) -> nx.DiGraph:
        """Extract passing network (subgraph of pass edges)"""
        subgraph = nx.DiGraph()
        
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == "pass":
                if successful_only and not data.get("successful", False):
                    continue
                
                # Filter by team if specified
                if team_id:
                    u_team = self.graph.nodes[u].get("team_id")
                    v_team = self.graph.nodes[v].get("team_id")
                    if u_team != team_id or v_team != team_id:
                        continue
                
                if not subgraph.has_edge(u, v):
                    subgraph.add_edge(u, v, weight=1, **data)
                else:
                    subgraph[u][v]["weight"] += 1
        
        return subgraph
    
    def get_pressing_network(self, team_id: Optional[int] = None) -> nx.DiGraph:
        """Extract pressing network (subgraph of pressure edges)"""
        subgraph = nx.DiGraph()
        
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == "pressure":
                # Filter by team if specified
                if team_id:
                    u_team = self.graph.nodes[u].get("team_id")
                    if u_team != team_id:
                        continue
                
                if not subgraph.has_edge(u, v):
                    subgraph.add_edge(u, v, weight=1, **data)
                else:
                    subgraph[u][v]["weight"] += 1
        
        return subgraph
    
    def get_passing_options_network(self, executed_only: bool = False) -> nx.DiGraph:
        """Extract passing options network"""
        subgraph = nx.DiGraph()
        
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == "passing_option":
                if executed_only and not data.get("executed", False):
                    continue
                
                if not subgraph.has_edge(u, v):
                    subgraph.add_edge(u, v, weight=1, **data)
                else:
                    subgraph[u][v]["weight"] += 1
        
        return subgraph
    
    def analyze_centrality(self, network_type: str = "passing") -> Dict:
        """Analyze node centrality in different networks"""
        if network_type == "passing":
            network = self.get_passing_network()
        elif network_type == "pressing":
            network = self.get_pressing_network()
        elif network_type == "options":
            network = self.get_passing_options_network()
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        return {
            "degree_centrality": nx.degree_centrality(network),
            "betweenness_centrality": nx.betweenness_centrality(network),
            "closeness_centrality": nx.closeness_centrality(network),
            "pagerank": nx.pagerank(network)
        }
    
    def to_dict(self) -> Dict:
        """Convert graph to dictionary representation"""
        return {
            "match_id": self.match_id,
            "nodes": dict(self.graph.nodes(data=True)),
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **data
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            "temporal_snapshots": self.temporal_snapshots
        }


def build_graph_from_dynamic_events(events_df: pd.DataFrame, match_id: int) -> GameGraph:
    """Build a game graph from SkillCorner dynamic events"""
    graph = GameGraph(match_id)
    
    for _, event in events_df.iterrows():
        timestamp = _time_to_seconds(event.get('time_start', '00:00.0'))
        event_type = event.get('event_type')
        player_id = event.get('player_id')
        team_id = event.get('team_id')
        
        if pd.isna(player_id) or pd.isna(team_id):
            continue
        
        # Add player node
        position = (event.get('x_start', 0), event.get('y_start', 0))
        graph.add_player_node(
            player_id=int(player_id),
            team_id=int(team_id),
            timestamp=timestamp,
            position=position,
            attributes={
                "phase_type": event.get('team_in_possession_phase_type'),
                "n_passing_options": event.get('n_passing_options', 0),
                "xthreat": event.get('xthreat')
            }
        )
        
        # Add edges based on event type
        if event_type == "player_possession":
            # Add passing option edges
            if event.get('n_passing_options', 0) > 0:
                # Note: In real implementation, you'd iterate through actual passing options
                pass
        
        elif event_type == "passing_option":
            target_player = event.get('player_targeted_id')
            if not pd.isna(target_player):
                executed = event.get('received', False)
                graph.add_passing_option_edge(
                    from_player=int(player_id),
                    to_player=int(target_player),
                    timestamp=timestamp,
                    executed=bool(executed),
                    attributes={
                        "xpass_completion": event.get('player_targeted_xpass_completion'),
                        "xthreat": event.get('player_targeted_xthreat'),
                        "dangerous": event.get('player_targeted_dangerous', False)
                    }
                )
        
        elif event_type == "on_ball_engagement":
            target_player = event.get('player_in_possession_id')
            if not pd.isna(target_player):
                graph.add_pressure_edge(
                    presser=int(player_id),
                    target=int(target_player),
                    timestamp=timestamp,
                    attributes={
                        "pressing_chain_id": event.get('pressing_chain'),
                        "pressing_chain_length": event.get('pressing_chain_length'),
                        "beaten_by_possession": event.get('beaten_by_possession', False)
                    }
                )
        
        elif event_type == "off_ball_run":
            end_position = (event.get('x_end', 0), event.get('y_end', 0))
            graph.add_off_ball_run_edge(
                runner=int(player_id),
                target_location=end_position,
                timestamp=timestamp,
                attributes={
                    "distance_covered": event.get('distance_covered'),
                    "speed_avg": event.get('speed_avg'),
                    "subtype": event.get('associated_off_ball_run_subtype')
                }
            )
    
    return graph


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
    
    # Build graph
    graph = build_graph_from_dynamic_events(events_df, match_id)
    
    print(f"Graph built: {graph.graph.number_of_nodes()} nodes, {graph.graph.number_of_edges()} edges")
    
    # Analyze passing network
    passing_network = graph.get_passing_network()
    print(f"\nPassing network: {passing_network.number_of_nodes()} nodes, {passing_network.number_of_edges()} edges")
    
    # Analyze centrality
    centrality = graph.analyze_centrality("passing")
    print(f"\nTop 5 players by PageRank:")
    top_players = sorted(centrality["pagerank"].items(), key=lambda x: x[1], reverse=True)[:5]
    for player, score in top_players:
        print(f"  {player}: {score:.4f}")

