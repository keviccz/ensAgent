"""
Dashboard components for EnsAgent.
Includes Overview and Spatial Analysis views.
"""
from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from streamlit_app.utils.state import get_state


# Soft chart palette matching the design spec
CHART_COLORS = ["#5A3683", "#6B8FD4", "#A3BFE8", "#D4E3F8", "#B0B1BB", "#E74C3C", "#34C759"]
CHART_FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"


def render_overview() -> None:
    """Render the Overview dashboard tab."""

    # Filter chips
    st.markdown(
        """
        <div class="ens-filters">
            <div class="ens-filter-chip">All Stages ▾</div>
            <div class="ens-filter-chip">Last 7 Days ▾</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI strip
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Data Status",
            value="Ready" if get_state("data_path") else "Not Set",
        )

    with col2:
        st.metric(
            label="Sample",
            value=get_state("sample_id") or "\u2014",
        )

    with col3:
        agents = get_state("agents", [])
        active_count = sum(1 for a in agents if a.status == "active")
        st.metric(
            label="Active Agents",
            value=f"{active_count}/{len(agents)}",
        )

    with col4:
        progress = get_state("pipeline_progress", 0.0)
        st.metric(
            label="Pipeline",
            value=f"{int(progress * 100)}%",
        )

    st.markdown('<div class="ens-spacer-md"></div>', unsafe_allow_html=True)

    # Two chart cards side-by-side
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        _render_pipeline_chart()

    with chart_col2:
        _render_agent_activity_chart()

    st.markdown('<div class="ens-spacer-md"></div>', unsafe_allow_html=True)

    # Quick actions
    st.markdown("### Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run End-to-End", use_container_width=True, type="secondary"):
            st.session_state["pending_action"] = "run_end_to_end"
            st.rerun()

    with col2:
        if st.button("Check Environments", use_container_width=True, type="secondary"):
            st.session_state["pending_action"] = "check_envs"
            st.rerun()

    with col3:
        if st.button("Show Config", use_container_width=True, type="secondary"):
            st.session_state["pending_action"] = "show_config"
            st.rerun()

    st.markdown('<div class="ens-spacer-md"></div>', unsafe_allow_html=True)

    # Pipeline stages
    st.markdown("### Pipeline Stages")

    pipeline_stage = get_state("pipeline_stage")

    stages = [
        ("1", "Tool Runner", "Spatial clustering methods (IRIS, BASS, GraphST, etc.)", "tool_runner"),
        ("2", "Scoring", "LLM-based domain evaluation and aggregation", "scoring"),
        ("3", "BEST Builder", "Select optimal domain labels per spot", "best"),
        ("4", "Annotation", "Multi-agent domain annotation with experts", "annotation"),
    ]

    for stage_num, name, desc, stage_key in stages:
        is_active = pipeline_stage == stage_key
        active_attr = 'data-active="true"' if is_active else ""
        st.markdown(
            f"""
            <div class="ens-stage" {active_attr}>
                <div class="ens-stage-icon">{stage_num}</div>
                <div class="ens-stage-body">
                    <div class="ens-stage-title">Stage {stage_num}: {name}</div>
                    <div class="ens-stage-desc">{desc}</div>
                </div>
                <div class="ens-stage-dot"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_pipeline_chart() -> None:
    """Render a pipeline stage progress bar chart."""
    st.markdown(
        """
        <div class="ens-card">
            <div class="ens-settings-card-title">Pipeline Progress</div>
            <div class="ens-settings-card-desc">Status of each pipeline stage</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stages = ["Tool Runner", "Scoring", "BEST Builder", "Annotation"]
    values = [0, 0, 0, 0]

    pipeline_stage = get_state("pipeline_stage")
    progress = get_state("pipeline_progress", 0.0)

    stage_map = {"tool_runner": 0, "scoring": 1, "best": 2, "annotation": 3}
    if pipeline_stage in stage_map:
        idx = stage_map[pipeline_stage]
        for i in range(idx):
            values[i] = 100
        values[idx] = int(progress * 100)

    df = pd.DataFrame({"Stage": stages, "Progress": values})

    try:
        import plotly.express as px

        fig = px.bar(
            df, x="Progress", y="Stage", orientation="h",
            color_discrete_sequence=["#A3BFE8"],
        )
        fig.update_layout(
            plot_bgcolor="#FEFEFE",
            paper_bgcolor="#FEFEFE",
            font_family=CHART_FONT,
            font_color="#181A1F",
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
            showlegend=False,
            xaxis=dict(
                range=[0, 100],
                showgrid=True,
                gridcolor="#E9E9EB",
                gridwidth=1,
                zeroline=False,
                ticksuffix="%",
            ),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(df.set_index("Stage"))


def _render_agent_activity_chart() -> None:
    """Render an agent activity line chart (placeholder with demo data)."""
    st.markdown(
        """
        <div class="ens-card">
            <div class="ens-settings-card-title">Agent Activity</div>
            <div class="ens-settings-card-desc">Recent agent operations over time</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    np.random.seed(7)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=7, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Tool Runner": np.random.randint(0, 8, 7),
        "Scoring": np.random.randint(0, 6, 7),
        "Annotation": np.random.randint(0, 5, 7),
    })

    try:
        import plotly.express as px

        df_melt = df.melt(id_vars="Date", var_name="Agent", value_name="Operations")
        fig = px.line(
            df_melt, x="Date", y="Operations", color="Agent",
            color_discrete_sequence=["#5A3683", "#6B8FD4", "#B0B1BB"],
        )
        fig.update_layout(
            plot_bgcolor="#FEFEFE",
            paper_bgcolor="#FEFEFE",
            font_family=CHART_FONT,
            font_color="#181A1F",
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.35,
                xanchor="center", x=0.5,
                font=dict(size=11),
            ),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#E9E9EB", gridwidth=1, zeroline=False),
        )
        fig.update_traces(line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(df.set_index("Date"))


def render_spatial_analysis() -> None:
    """Render the Spatial Analysis tab with visualizations."""
    spatial_data = get_state("spatial_data")

    if spatial_data is not None:
        _render_spatial_plot(spatial_data)
    else:
        _render_demo_spatial_plot()


def _render_demo_spatial_plot() -> None:
    """Render a demo spatial plot with synthetic data."""
    st.markdown(
        """
        <div class="ens-alert">
            <div class="ens-alert-header">
                <span class="ens-alert-icon">Demo</span>
                <span class="ens-alert-title">Demo Visualization</span>
            </div>
            <p class="ens-alert-text">
                This is <strong>synthetic demo data</strong> (randomly generated) to showcase visualization capabilities.
                Run the pipeline with your actual data to see real spatial transcriptomics results here.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Generate synthetic spatial data
    np.random.seed(42)
    n_spots = 500

    centers = np.array([
        [2, 2], [8, 2], [5, 5], [2, 8], [8, 8], [5, 2], [5, 8]
    ])

    spots_per_cluster = n_spots // len(centers)
    coords = []
    labels = []

    for i, center in enumerate(centers):
        cluster_coords = np.random.normal(loc=center, scale=0.8, size=(spots_per_cluster, 2))
        coords.append(cluster_coords)
        labels.extend([f"Domain {i+1}"] * spots_per_cluster)

    coords = np.vstack(coords)

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Domain": labels,
        "Expression": np.random.exponential(2, len(labels)),
    })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Domain Clustering")
        try:
            import plotly.express as px

            fig = px.scatter(
                df, x="x", y="y", color="Domain",
                color_discrete_sequence=CHART_COLORS,
                height=400,
            )
            fig.update_layout(
                plot_bgcolor="#FEFEFE",
                paper_bgcolor="#FEFEFE",
                font_family=CHART_FONT,
                font_color="#181A1F",
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.98,
                    xanchor="right",
                    x=0.98,
                    bgcolor="rgba(255,255,255,0.78)",
                    bordercolor="#E9E9EB",
                    borderwidth=1,
                    font=dict(size=10),
                ),
            )
            fig.update_traces(marker=dict(size=6, opacity=0.8))
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.scatter_chart(df, x="x", y="y", color="Domain", height=400)

    with col2:
        st.markdown("#### Expression Levels")
        try:
            import plotly.express as px

            fig = px.scatter(
                df, x="x", y="y", color="Expression",
                color_continuous_scale=["#D4E3F8", "#6B8FD4", "#5A3683"],
                height=400,
            )
            fig.update_layout(
                plot_bgcolor="#FEFEFE",
                paper_bgcolor="#FEFEFE",
                font_family=CHART_FONT,
                font_color="#181A1F",
                margin=dict(l=20, r=20, t=20, b=20),
                coloraxis_colorbar=dict(title="", thickness=12, len=0.9),
            )
            fig.update_traces(marker=dict(size=6, opacity=0.8))
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.scatter_chart(df, x="x", y="y", color="Expression", height=400)

    st.markdown("#### Statistics")

    stats_cols = st.columns(4)
    with stats_cols[0]:
        st.metric("Total Spots", n_spots)
    with stats_cols[1]:
        st.metric("Domains", len(centers))
    with stats_cols[2]:
        st.metric("Avg Expression", f"{df['Expression'].mean():.2f}")
    with stats_cols[3]:
        st.metric("Coverage", "100%")


def _render_spatial_plot(data: pd.DataFrame) -> None:
    """Render spatial plot with actual data."""
    try:
        import plotly.express as px

        x_col = "x" if "x" in data.columns else data.columns[0]
        y_col = "y" if "y" in data.columns else data.columns[1]
        color_col = None
        for c in ["Domain", "domain", "cluster", "label", "annotation"]:
            if c in data.columns:
                color_col = c
                break

        fig = px.scatter(
            data, x=x_col, y=y_col, color=color_col,
            color_discrete_sequence=CHART_COLORS,
            height=500,
        )
        fig.update_layout(
            plot_bgcolor="#FEFEFE",
            paper_bgcolor="#FEFEFE",
            font_family=CHART_FONT,
            font_color="#181A1F",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.dataframe(data.head(100))
        st.warning("Install plotly for better visualizations: pip install plotly")
