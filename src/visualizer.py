"""
Visualization functions for phase of play transition analysis.
Uses mplsoccer, matplotlib, NetworkX, and plotly for various visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx
from mplsoccer import Pitch, VerticalPitch
from typing import Dict, List, Optional, Tuple

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_transition_network(
    edges: Dict[Tuple[str, str], int],
    nodes: List[str],
    figsize: Tuple[int, int] = (12, 8),
    node_size: int = 3000,
    font_size: int = 10,
) -> plt.Figure:
    """
    Plot transition network graph using NetworkX.

    Args:
        edges: Dictionary of edges with (from, to) keys and count values
        nodes: List of node names
        figsize: Figure size
        node_size: Size of nodes
        font_size: Font size for labels

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    # Add edges with weights
    for (from_node, to_node), weight in edges.items():
        G.add_edge(from_node, to_node, weight=weight)

    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color="lightblue", node_size=node_size, alpha=0.9, ax=ax
    )

    # Draw edges with width proportional to weight
    edge_widths = [G[u][v]["weight"] * 0.5 for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.6,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        ax=ax,
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)

    # Add edge labels (transition counts)
    edge_labels = {(u, v): str(d["weight"]) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    ax.set_title("Phase Transition Network", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    return fig


def plot_transition_heatmap(
    df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot heatmap of transition locations on pitch.

    Args:
        df: DataFrame with transition data including x_start, y_start, x_end, y_end
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Create pitch with explicit dimensions for SkillCorner data
    # SkillCorner uses meters with center at (0,0), standard pitch is 105m x 68m
    # Try skillcorner pitch type first, fallback to custom if not supported
    try:
        pitch = Pitch(
            pitch_type="skillcorner",
            pitch_length=105,
            pitch_width=68,
            line_color="black",
            line_zorder=2,
            line_alpha=0.75,
        )
    except (TypeError, ValueError, AttributeError):
        # Fallback: create custom pitch with dimensions only
        # Some mplsoccer versions may not support skillcorner type
        pitch = Pitch(
            pitch_length=105,
            pitch_width=68,
            line_color="black",
            line_zorder=2,
            line_alpha=0.75,
        )
    fig, ax = pitch.draw(figsize=figsize)

    # Filter transitions that lead to goals
    goal_transitions = df[(df["transition"].notna()) & (df["leads_to_goal"])]

    # Plot transition start locations
    if not goal_transitions.empty:
        ax.scatter(
            goal_transitions["x_start"],
            goal_transitions["y_start"],
            s=100,
            alpha=0.6,
            color="green",
            label="Goal transitions (start)",
            zorder=3,
        )
        ax.scatter(
            goal_transitions["x_end"],
            goal_transitions["y_end"],
            s=100,
            alpha=0.6,
            color="red",
            label="Goal transitions (end)",
            zorder=3,
        )

    # Plot all transitions
    all_transitions = df[df["transition"].notna()]
    if not all_transitions.empty:
        ax.scatter(
            all_transitions["x_start"],
            all_transitions["y_start"],
            s=20,
            alpha=0.2,
            color="gray",
            label="All transitions",
            zorder=2,
        )

    ax.set_title("Transition Locations on Pitch", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")

    plt.tight_layout()
    return fig


def plot_transition_success_rates(
    metrics_df: pd.DataFrame, top_n: int = 10, figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot bar chart of transition success rates.

    Args:
        metrics_df: DataFrame with transition success metrics
        top_n: Number of top transitions to show
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get top N transitions by goal rate
    top_transitions = metrics_df.nlargest(top_n, "goal_rate")

    x_pos = np.arange(len(top_transitions))
    width = 0.35

    # Create bars
    bars1 = ax.bar(
        x_pos - width / 2,
        top_transitions["goal_rate"],
        width,
        label="Goal Rate (%)",
        color="green",
        alpha=0.7,
    )
    bars2 = ax.bar(
        x_pos + width / 2,
        top_transitions["shot_rate"],
        width,
        label="Shot Rate (%)",
        color="orange",
        alpha=0.7,
    )

    ax.set_xlabel("Transition", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title(
        f"Top {top_n} Transitions by Success Rate", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_transitions["transition"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sankey_diagram(
    edges: Dict[Tuple[str, str], int], figsize: Tuple[int, int] = (12, 8)
):
    """
    Create Sankey diagram of phase transitions using Plotly.

    Args:
        edges: Dictionary of edges with (from, to) keys and count values
        figsize: Figure size (not used for Plotly, but kept for consistency)

    Returns:
        Plotly figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for Sankey diagrams. Install it with: pip install plotly"
        )

    # Get all unique nodes
    nodes_set = set()
    for from_node, to_node in edges.keys():
        nodes_set.add(from_node)
        nodes_set.add(to_node)
    nodes_list = sorted(list(nodes_set))

    # Create node mapping
    node_map = {node: i for i, node in enumerate(nodes_list)}

    # Create source, target, and value lists
    source = []
    target = []
    value = []

    for (from_node, to_node), count in edges.items():
        source.append(node_map[from_node])
        target.append(node_map[to_node])
        value.append(count)

    # Create Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes_list,
                    color="lightblue",
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color="rgba(0, 100, 200, 0.4)",
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Phase Transition Flow (Sankey Diagram)", font_size=12, height=600
    )

    return fig


def plot_phase_distribution(
    df: pd.DataFrame,
    group_by: str = "team_in_possession_shortname",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot phase type distribution by group.

    Args:
        df: DataFrame with phase data
        group_by: Column to group by
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Count phases by type and group
    phase_counts = (
        df.groupby([group_by, "team_in_possession_phase_type"])
        .size()
        .unstack(fill_value=0)
    )

    # Calculate percentages
    phase_percentages = phase_counts.div(phase_counts.sum(axis=1), axis=0) * 100

    # Plot stacked bar chart
    phase_percentages.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")

    ax.set_xlabel(group_by.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Phase Type Distribution", fontsize=14, fontweight="bold")
    ax.legend(title="Phase Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_transition_timeline(
    df: pd.DataFrame, match_id: Optional[int] = None, figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot timeline of transitions throughout the match.

    Args:
        df: DataFrame with transition data
        match_id: Optional match ID to filter by
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if match_id is not None:
        df = df[df["match_id"] == match_id]

    # Get transitions with timing
    transitions = df[df["transition"].notna()].copy()

    if transitions.empty:
        ax.text(
            0.5,
            0.5,
            "No transitions found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        return fig

    # Create timeline
    for idx, row in transitions.iterrows():
        minute = row["minute_float"]
        period = row["period"]
        x_pos = minute if period == 1 else minute + 45  # Approximate

        color = (
            "green"
            if row["leads_to_goal"]
            else "orange" if row["leads_to_shot"] else "gray"
        )
        alpha = 1.0 if row["leads_to_goal"] else 0.6 if row["leads_to_shot"] else 0.3

        ax.scatter(x_pos, 0, s=100, color=color, alpha=alpha, zorder=3)

    # Add period separator
    ax.axvline(
        x=45, color="red", linestyle="--", linewidth=2, label="Half Time", zorder=2
    )

    ax.set_xlabel("Minute", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    ax.set_title("Transition Timeline", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])

    # Add legend
    goal_patch = mpatches.Patch(color="green", label="Goal")
    shot_patch = mpatches.Patch(color="orange", label="Shot")
    other_patch = mpatches.Patch(color="gray", label="Other")
    ax.legend(handles=[goal_patch, shot_patch, other_patch], loc="upper right")

    plt.tight_layout()
    return fig


def plot_spatial_transition_heatmap(
    df: pd.DataFrame, outcome: str = "goal", figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot heatmap of transition start locations by outcome.

    Args:
        df: DataFrame with transition data
        outcome: Outcome to filter by ('goal', 'shot', 'all')
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Create pitch with explicit dimensions for SkillCorner data
    # SkillCorner uses meters with center at (0,0), standard pitch is 105m x 68m
    # Try skillcorner pitch type first, fallback to custom if not supported
    try:
        pitch = Pitch(
            pitch_type="skillcorner",
            pitch_length=105,
            pitch_width=68,
            line_color="black",
            line_zorder=2,
            line_alpha=0.75,
        )
    except (TypeError, ValueError, AttributeError):
        # Fallback: create custom pitch with dimensions only
        # Some mplsoccer versions may not support skillcorner type
        pitch = Pitch(
            pitch_length=105,
            pitch_width=68,
            line_color="black",
            line_zorder=2,
            line_alpha=0.75,
        )
    fig, ax = pitch.draw(figsize=figsize)

    # Filter transitions
    transitions = df[df["transition"].notna()].copy()

    if outcome == "goal":
        transitions = transitions[transitions["leads_to_goal"]]
    elif outcome == "shot":
        transitions = transitions[transitions["leads_to_shot"]]

    if transitions.empty:
        ax.text(
            0.5,
            0.5,
            f"No {outcome} transitions found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        return fig

    # Create 2D histogram
    x_bins = np.linspace(-52.5, 52.5, 21)  # Pitch length
    y_bins = np.linspace(-34, 34, 17)  # Pitch width

    H, xedges, yedges = np.histogram2d(
        transitions["x_start"], transitions["y_start"], bins=[x_bins, y_bins]
    )

    # Plot heatmap
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[-52.5, 52.5, -34, 34],
        cmap="YlOrRd",
        alpha=0.6,
        aspect="auto",
        zorder=1,
    )

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Transition Count")

    ax.set_title(
        f"Transition Start Locations ({outcome.title()})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def plot_sequence_analysis(
    sequences_df: pd.DataFrame, top_n: int = 10, figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot most common phase sequences.

    Args:
        sequences_df: DataFrame with sequence information
        top_n: Number of top sequences to show
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get top sequences
    top_sequences = sequences_df.nlargest(top_n, "count")

    y_pos = np.arange(len(top_sequences))

    # Create horizontal bar chart
    bars = ax.barh(y_pos, top_sequences["count"], alpha=0.7)

    # Color bars by goal rate
    colors = ["green" if rate > 0 else "gray" for rate in top_sequences["goal_rate"]]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_sequences["sequence"], fontsize=9)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title(f"Top {top_n} Phase Sequences", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_team_comparison(
    team_metrics_df: pd.DataFrame,
    metric: str = "goal_rate",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot team comparison for a specific metric.

    Args:
        team_metrics_df: DataFrame with team metrics
        metric: Metric to compare
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Aggregate by team if needed
    if "team" in team_metrics_df.columns:
        team_agg = (
            team_metrics_df.groupby("team")[metric].mean().sort_values(ascending=False)
        )
    else:
        team_agg = team_metrics_df.set_index("team_in_possession_shortname")[
            metric
        ].sort_values(ascending=False)

    # Plot bar chart
    bars = ax.bar(range(len(team_agg)), team_agg.values, alpha=0.7, color="steelblue")

    ax.set_xticks(range(len(team_agg)))
    ax.set_xticklabels(team_agg.index, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(
        f'Team Comparison: {metric.replace("_", " ").title()}',
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig
