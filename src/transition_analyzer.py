"""
Core transition analysis logic for phases of play.
Provides functions to analyze transitions, calculate metrics, and identify patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter


def get_transition_counts(df: pd.DataFrame, 
                         group_by: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Count transitions by type.
    
    Args:
        df: Phases DataFrame with transition information
        group_by: Optional list of columns to group by (e.g., ['team_in_possession_shortname'])
        
    Returns:
        DataFrame with transition counts
    """
    transitions = df[df['transition'].notna()].copy()
    
    if group_by is None:
        group_by = []
    
    counts = (
        transitions.groupby(group_by + ['transition'])
        .size()
        .reset_index(name='count')
    )
    
    return counts.sort_values('count', ascending=False)


def get_transition_success_rates(df: pd.DataFrame,
                                 group_by: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate success rates (goals/shots) for each transition type.
    
    Args:
        df: Phases DataFrame with transition information
        group_by: Optional list of columns to group by
        
    Returns:
        DataFrame with transition success metrics
    """
    transitions = df[df['transition'].notna()].copy()
    
    if group_by is None:
        group_by = []
    
    metrics = transitions.groupby(group_by + ['transition']).agg({
        'transition_leads_to_goal': 'sum',
        'transition_leads_to_shot': 'sum',
        'leads_to_goal': 'sum',
        'leads_to_shot': 'sum',
        'transition': 'count'
    }).rename(columns={
        'transition': 'total_transitions',
        'leads_to_goal': 'total_goals',
        'leads_to_shot': 'total_shots'
    }).reset_index()
    
    metrics['goals_per_transition'] = metrics['transition_leads_to_goal'] / metrics['total_transitions']
    metrics['shots_per_transition'] = metrics['transition_leads_to_shot'] / metrics['total_transitions']
    metrics['goal_rate'] = metrics['transition_leads_to_goal'] / metrics['total_transitions'] * 100
    metrics['shot_rate'] = metrics['transition_leads_to_shot'] / metrics['total_transitions'] * 100
    
    return metrics.sort_values('goal_rate', ascending=False)


def get_transition_network(df: pd.DataFrame,
                          min_count: int = 1) -> Tuple[Dict, List]:
    """
    Build transition network graph structure.
    
    Args:
        df: Phases DataFrame with transition information
        min_count: Minimum count for a transition to be included
        
    Returns:
        Tuple of (edges dictionary with counts, list of nodes)
    """
    transitions = df[df['transition'].notna()].copy()
    
    # Parse transitions into from -> to pairs
    edges = {}
    nodes = set()
    
    for transition in transitions['transition']:
        if ' -> ' in str(transition):
            from_phase, to_phase = transition.split(' -> ')
            nodes.add(from_phase)
            nodes.add(to_phase)
            
            edge_key = (from_phase, to_phase)
            edges[edge_key] = edges.get(edge_key, 0) + 1
    
    # Filter by minimum count
    edges = {k: v for k, v in edges.items() if v >= min_count}
    
    # Update nodes to only include those in filtered edges
    nodes = set()
    for from_node, to_node in edges.keys():
        nodes.add(from_node)
        nodes.add(to_node)
    
    return edges, list(nodes)


def get_phase_durations(df: pd.DataFrame,
                       group_by: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate average phase durations by phase type.
    
    Args:
        df: Phases DataFrame
        group_by: Optional list of columns to group by
        
    Returns:
        DataFrame with duration metrics
    """
    if group_by is None:
        group_by = []
    
    duration_metrics = df.groupby(group_by + ['team_in_possession_phase_type']).agg({
        'duration': ['mean', 'median', 'std', 'count'],
        'leads_to_goal': 'sum',
        'leads_to_shot': 'sum'
    }).reset_index()
    
    # Flatten column names
    duration_metrics.columns = [
        '_'.join(col).strip('_') if col[1] else col[0] 
        for col in duration_metrics.columns.values
    ]
    
    return duration_metrics


def get_spatial_transition_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze transitions by spatial location (thirds, channels).
    
    Args:
        df: Phases DataFrame with transition information
        
    Returns:
        DataFrame with spatial transition metrics
    """
    transitions = df[df['transition'].notna()].copy()
    
    spatial_metrics = transitions.groupby([
        'third_start', 'channel_start', 'third_end', 'channel_end', 'transition'
    ]).agg({
        'transition_leads_to_goal': 'sum',
        'transition_leads_to_shot': 'sum',
        'transition': 'count',
        'x_start': 'mean',
        'y_start': 'mean',
        'x_end': 'mean',
        'y_end': 'mean'
    }).rename(columns={
        'transition': 'count',
        'x_start': 'avg_x_start',
        'y_start': 'avg_y_start',
        'x_end': 'avg_x_end',
        'y_end': 'avg_y_end'
    }).reset_index()
    
    spatial_metrics['goal_rate'] = (
        spatial_metrics['transition_leads_to_goal'] / 
        spatial_metrics['count'] * 100
    )
    
    return spatial_metrics.sort_values('goal_rate', ascending=False)


def get_possession_sequences(df: pd.DataFrame,
                            min_phases: int = 2) -> pd.DataFrame:
    """
    Extract and analyze possession sequences.
    
    Args:
        df: Phases DataFrame with possession sequences
        min_phases: Minimum number of phases in a sequence to include
        
    Returns:
        DataFrame with sequence information
    """
    sequences = []
    
    for (match_id, period, seq_id), group in df.groupby(
        ['match_id', 'period', 'possession_sequence']
    ):
        if len(group) >= min_phases:
            sequence_phases = group['team_in_possession_phase_type'].tolist()
            sequence_str = ' -> '.join(sequence_phases)
            
            sequences.append({
                'match_id': match_id,
                'period': period,
                'possession_sequence': seq_id,
                'team': group['team_in_possession_shortname'].iloc[0],
                'sequence': sequence_str,
                'num_phases': len(group),
                'duration': group['duration'].sum(),
                'leads_to_goal': group['leads_to_goal'].any(),
                'leads_to_shot': group['leads_to_shot'].any(),
                'x_start': group['x_start'].iloc[0],
                'y_start': group['y_start'].iloc[0],
                'x_end': group['x_end'].iloc[-1],
                'y_end': group['y_end'].iloc[-1],
            })
    
    return pd.DataFrame(sequences)


def get_common_sequences(df: pd.DataFrame,
                        min_count: int = 2) -> pd.DataFrame:
    """
    Find most common phase sequences.
    
    Args:
        df: Phases DataFrame with possession sequences
        min_count: Minimum occurrence count
        
    Returns:
        DataFrame with common sequences and their counts
    """
    sequences_df = get_possession_sequences(df, min_phases=2)
    
    if sequences_df.empty:
        return pd.DataFrame()
    
    sequence_counts = sequences_df.groupby('sequence').agg({
        'sequence': 'count',
        'leads_to_goal': 'sum',
        'leads_to_shot': 'sum'
    }).rename(columns={'sequence': 'count'}).reset_index()
    
    sequence_counts = sequence_counts[sequence_counts['count'] >= min_count]
    sequence_counts['goal_rate'] = (
        sequence_counts['leads_to_goal'] / sequence_counts['count'] * 100
    )
    sequence_counts['shot_rate'] = (
        sequence_counts['leads_to_shot'] / sequence_counts['count'] * 100
    )
    
    return sequence_counts.sort_values('count', ascending=False)


def get_team_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare transition patterns between teams.
    
    Args:
        df: Phases DataFrame
        
    Returns:
        DataFrame with team comparison metrics
    """
    team_metrics = df.groupby('team_in_possession_shortname').agg({
        'team_in_possession_phase_type': lambda x: x.value_counts().to_dict(),
        'transition': lambda x: x[x.notna()].count(),
        'leads_to_goal': 'sum',
        'leads_to_shot': 'sum',
        'duration': 'mean',
        'possession_sequence': 'nunique'
    }).reset_index()
    
    # Calculate phase type distributions
    phase_distributions = []
    for _, row in team_metrics.iterrows():
        team = row['team_in_possession_shortname']
        phase_counts = row['team_in_possession_phase_type']
        
        total_phases = sum(phase_counts.values()) if isinstance(phase_counts, dict) else 0
        
        for phase_type, count in (phase_counts.items() if isinstance(phase_counts, dict) else []):
            phase_distributions.append({
                'team': team,
                'phase_type': phase_type,
                'count': count,
                'percentage': (count / total_phases * 100) if total_phases > 0 else 0
            })
    
    return pd.DataFrame(phase_distributions)


def get_period_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare transition patterns between periods.
    
    Args:
        df: Phases DataFrame
        
    Returns:
        DataFrame with period comparison metrics
    """
    period_metrics = df.groupby('period').agg({
        'transition': lambda x: x[x.notna()].count(),
        'leads_to_goal': 'sum',
        'leads_to_shot': 'sum',
        'duration': 'mean',
        'possession_sequence': 'nunique'
    }).reset_index()
    
    period_metrics['goals_per_possession'] = (
        period_metrics['leads_to_goal'] / period_metrics['possession_sequence']
    )
    period_metrics['shots_per_possession'] = (
        period_metrics['leads_to_shot'] / period_metrics['possession_sequence']
    )
    
    return period_metrics


def filter_transitions(df: pd.DataFrame,
                      phase_types: Optional[List[str]] = None,
                      periods: Optional[List[int]] = None,
                      outcomes: Optional[List[str]] = None,
                      teams: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter transitions by various criteria.
    
    Args:
        df: Phases DataFrame
        phase_types: List of phase types to include
        periods: List of periods to include
        outcomes: List of outcomes to include ('goal', 'shot', 'no_outcome')
        teams: List of team names to include
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if phase_types is not None:
        filtered_df = filtered_df[
            filtered_df['team_in_possession_phase_type'].isin(phase_types)
        ]
    
    if periods is not None:
        filtered_df = filtered_df[filtered_df['period'].isin(periods)]
    
    if outcomes is not None:
        filtered_df = filtered_df[filtered_df['outcome'].isin(outcomes)]
    
    if teams is not None:
        filtered_df = filtered_df[
            filtered_df['team_in_possession_shortname'].isin(teams)
        ]
    
    return filtered_df

