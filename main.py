"""
Streamlit web application for Phase of Play Transition Analyzer.
Provides interactive dashboard for analyzing phase transitions in football matches.
"""

import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Phase of Play Transition Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.data_loader import (
    load_matches_metadata,
    load_and_preprocess_match,
    load_multiple_matches,
)
from src.transition_analyzer import (
    get_transition_counts,
    get_transition_success_rates,
    get_transition_network,
    get_common_sequences,
    get_team_comparison,
    filter_transitions,
)
from src.visualizer import (
    plot_transition_network,
    plot_transition_heatmap,
    plot_transition_success_rates,
    plot_sankey_diagram,
    plot_phase_distribution,
    plot_spatial_transition_heatmap,
    plot_transition_timeline,
)

# Title
st.title("⚽ Phase of Play Transition Analyzer")
st.markdown("Analyze tactical patterns and transition sequences in football matches")

# Sidebar for filters
st.sidebar.header("Filters")


# Load matches metadata
@st.cache_data
def load_matches():
    """Load matches metadata with caching."""
    return load_matches_metadata()


matches_df = load_matches()

# Match selection
match_options = {
    f"{row['home_team']['short_name']} vs {row['away_team']['short_name']} ({row['date_time'][:10]})": row[
        "id"
    ]
    for _, row in matches_df.iterrows()
}

selected_match_name = st.sidebar.selectbox(
    "Select Match", options=list(match_options.keys()), index=0
)

selected_match_id = match_options[selected_match_name]

# Multi-match analysis option
multi_match = st.sidebar.checkbox("Analyze Multiple Matches", value=False)

if multi_match:
    match_ids = st.sidebar.multiselect(
        "Select Matches",
        options=list(match_options.values()),
        default=[selected_match_id],
    )
else:
    match_ids = [selected_match_id]


# Load data
@st.cache_data
def load_phases_data(match_ids):
    """Load phases data with caching."""
    return load_multiple_matches(match_ids)


if match_ids:
    with st.spinner("Loading data..."):
        phases_df = load_phases_data(match_ids)

    if phases_df.empty:
        st.error("No data available for selected matches.")
        st.stop()

    # Additional filters
    st.sidebar.subheader("Additional Filters")

    # Phase type filter
    phase_types = phases_df["team_in_possession_phase_type"].unique().tolist()
    selected_phases = st.sidebar.multiselect(
        "Phase Types", options=phase_types, default=phase_types
    )

    # Period filter
    periods = phases_df["period"].unique().tolist()
    selected_periods = st.sidebar.multiselect(
        "Periods", options=periods, default=periods
    )

    # Outcome filter
    outcomes = ["goal", "shot", "no_outcome"]
    selected_outcomes = st.sidebar.multiselect(
        "Outcomes", options=outcomes, default=outcomes
    )

    # Team filter
    teams = phases_df["team_in_possession_shortname"].unique().tolist()
    selected_teams = st.sidebar.multiselect("Teams", options=teams, default=teams)

    # Apply filters
    filtered_df = filter_transitions(
        phases_df,
        phase_types=selected_phases if selected_phases else None,
        periods=selected_periods if selected_periods else None,
        outcomes=selected_outcomes if selected_outcomes else None,
        teams=selected_teams if selected_teams else None,
    )

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Overview",
            "Transition Network",
            "Spatial Analysis",
            "Team Comparison",
            "Sequences",
        ]
    )

    with tab1:
        st.header("Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Phases", len(filtered_df))

        with col2:
            transitions_count = filtered_df["transition"].notna().sum()
            st.metric("Total Transitions", transitions_count)

        with col3:
            goals = filtered_df["leads_to_goal"].sum()
            st.metric("Goals", goals)

        with col4:
            shots = filtered_df["leads_to_shot"].sum()
            st.metric("Shots", shots)

        # Transition counts
        st.subheader("Most Common Transitions")
        transition_counts = get_transition_counts(filtered_df)
        st.dataframe(transition_counts.head(10), use_container_width=True)

        # Success rates
        st.subheader("Transition Success Rates")
        success_rates = get_transition_success_rates(filtered_df)
        st.dataframe(
            success_rates[
                ["transition", "total_transitions", "goal_rate", "shot_rate"]
            ].head(10),
            use_container_width=True,
        )

        # Success rates chart
        fig = plot_transition_success_rates(success_rates, top_n=10, figsize=(10, 6))
        st.pyplot(fig)

    with tab2:
        st.header("Transition Network")

        # Network visualization
        min_count = st.slider("Minimum Transition Count", 1, 20, 5)
        edges, nodes = get_transition_network(filtered_df, min_count=min_count)

        if edges:
            fig = plot_transition_network(edges, nodes, figsize=(12, 8))
            st.pyplot(fig)

            # Sankey diagram
            st.subheader("Sankey Diagram")
            sankey_fig = plot_sankey_diagram(edges)
            st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.info("No transitions found with the selected criteria.")

    with tab3:
        st.header("Spatial Analysis")

        # Transition heatmap
        st.subheader("Transition Locations")
        fig = plot_transition_heatmap(filtered_df, figsize=(12, 8))
        st.pyplot(fig)

        # Outcome-specific heatmaps
        outcome_option = st.selectbox(
            "Select Outcome", options=["all", "goal", "shot"], index=0
        )

        if outcome_option != "all":
            st.subheader(f"{outcome_option.title()} Transitions Heatmap")
            fig = plot_spatial_transition_heatmap(
                filtered_df, outcome=outcome_option, figsize=(12, 8)
            )
            st.pyplot(fig)

        # Timeline
        if len(match_ids) == 1:
            st.subheader("Transition Timeline")
            fig = plot_transition_timeline(
                filtered_df, match_id=match_ids[0], figsize=(14, 6)
            )
            st.pyplot(fig)

    with tab4:
        st.header("Team Comparison")

        # Phase distribution
        st.subheader("Phase Type Distribution")
        fig = plot_phase_distribution(
            filtered_df, group_by="team_in_possession_shortname", figsize=(12, 6)
        )
        st.pyplot(fig)

        # Team-specific success rates
        st.subheader("Team Transition Success Rates")
        team_success = get_transition_success_rates(
            filtered_df, group_by=["team_in_possession_shortname"]
        )
        st.dataframe(
            team_success[
                [
                    "team_in_possession_shortname",
                    "transition",
                    "goal_rate",
                    "total_transitions",
                ]
            ].head(20),
            use_container_width=True,
        )

        # Team comparison table
        st.subheader("Team Metrics")
        team_comparison = get_team_comparison(filtered_df)
        st.dataframe(team_comparison.head(30), use_container_width=True)

    with tab5:
        st.header("Phase Sequences")

        # Common sequences
        st.subheader("Most Common Sequences")
        min_count = st.slider("Minimum Sequence Count", 1, 10, 3, key="seq_count")
        common_sequences = get_common_sequences(filtered_df, min_count=min_count)

        if not common_sequences.empty:
            st.dataframe(common_sequences.head(15), use_container_width=True)
        else:
            st.info("No sequences found with the selected criteria.")

    # Export options
    st.sidebar.header("Export")

    if st.sidebar.button("Export Transition Data"):
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="transition_data.csv",
            mime="text/csv",
        )

else:
    st.info("Please select at least one match to analyze.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Phase of Play Transition Analyzer**")
st.sidebar.markdown("Built with Streamlit and SkillCorner Open Data")
