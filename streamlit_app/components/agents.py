"""
Agent status and orchestrator components for EnsAgent.
"""
from __future__ import annotations

import html
import streamlit as st
from typing import List

from streamlit_app.utils.state import get_state, AgentStatus
from streamlit_app.styles import get_agent_card


def render_agent_status(agent: AgentStatus) -> None:
    """Render a single agent status card."""
    status_class = agent.status
    if status_class == "completed":
        status_class = "active"

    st.markdown(
        get_agent_card(
            icon=agent.icon,
            name=agent.name,
            status=status_class,
            description=agent.description,
        ),
        unsafe_allow_html=True,
    )


def render_agent_orchestrator() -> None:
    """Render the agent orchestrator dashboard."""
    agents: List[AgentStatus] = get_state("agents", [])

    if not agents:
        st.info("No agents configured.")
        return

    cols = st.columns(2)

    for i, agent in enumerate(agents):
        with cols[i % 2]:
            _render_agent_card_enhanced(agent)

    st.markdown('<div class="ens-spacer-sm"></div>', unsafe_allow_html=True)

    # Activity log
    st.markdown("### Activity Log")

    pipeline_logs = get_state("pipeline_logs", [])

    if pipeline_logs:
        for log in reversed(pipeline_logs[-10:]):
            timestamp = log["timestamp"].strftime("%H:%M:%S")
            raw_level = str(log.get("level", "info")).lower()
            safe_level = raw_level if raw_level in {"info", "warning", "error", "success"} else "info"
            safe_message = html.escape(str(log.get("message", "")), quote=False).replace("\n", "<br/>")

            st.markdown(
                f"""
                <div class="ens-log-row">
                    <span class="ens-log-time">{timestamp}</span>
                    <span class="ens-log-message" data-level="{safe_level}">
                        {safe_message}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="ens-empty-state ens-empty-state-sm">
                No activity yet. Start the pipeline to see agent activity.
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_agent_card_enhanced(agent: AgentStatus) -> None:
    """Render an enhanced agent card with progress."""
    status_dot_class = "active" if agent.status == "completed" else agent.status
    safe_icon = html.escape(str(agent.icon), quote=False)
    safe_name = html.escape(str(agent.name), quote=False)
    safe_desc = html.escape(str(agent.description), quote=False)
    safe_status = html.escape(str(agent.status).upper(), quote=False)

    progress_html = ""
    if agent.status == "active" and agent.progress > 0:
        progress_html = f"""
            <div class="ens-agent-progress">
                <div class="ens-agent-progress-bar" style="--progress: {agent.progress * 100}%;"></div>
            </div>
        """

    st.markdown(
        f"""
        <div class="ens-agent-card" data-status="{agent.status}">
            <div class="ens-agent-icon">{safe_icon}</div>
            <div class="ens-agent-meta">
                <div class="ens-agent-title">
                    <span class="status-dot {status_dot_class}"></span>
                    {safe_name}
                </div>
                <div class="ens-agent-desc">{safe_desc}</div>
            </div>
            <div class="ens-agent-status">{safe_status}</div>
            {progress_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
