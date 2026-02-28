"""
EnsAgent Streamlit Application
==============================
Ensemble Multi-Agent System for Spatial Transcriptomics Analysis

Run with:
    streamlit run streamlit_app/main.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="EnsAgent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/ensagent/ensagent",
        "Report a bug": "https://github.com/ensagent/ensagent/issues",
        "About": "EnsAgent: Ensemble Multi-Agent System for Spatial Transcriptomics",
    },
)

import sys
from pathlib import Path
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from streamlit_app.styles import get_premium_css
from streamlit_app.utils.state import init_session_state, get_state
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.dashboard import render_spatial_analysis
from streamlit_app.components.agents import render_agent_orchestrator
from streamlit_app.components.chat import render_chat_interface, process_pending_response
from streamlit_app.components.settings import render_settings


PAGE_TITLES = {
    "chat": ("Conversation", "Ask questions or give commands to run the analysis pipeline."),
    "spatial": ("Analysis", "Visualize spatial transcriptomics data and clustering results."),
    "agents": ("Agent Orchestrator", "Monitor the status of all sub-agents in the EnsAgent system."),
    "settings": ("Settings", "Configure API credentials, pipeline parameters, and preferences."),
}


def main():
    """Main application entry point."""
    init_session_state()
    process_pending_response()
    st.markdown(get_premium_css(), unsafe_allow_html=True)
    render_sidebar()
    render_main_content()


def render_main_content():
    """Render the main content area with page routing."""
    active_page = get_state("active_page", "chat")

    if active_page not in {"chat", "settings"}:
        title, subtitle = PAGE_TITLES.get(active_page, PAGE_TITLES["chat"])
        st.markdown(
            f"""
            <div class="ens-header">
                <div class="ens-page-title">{title}</div>
                <p class="ens-page-subtitle">{subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if active_page == "chat":
        render_chat_interface()
    elif active_page == "spatial":
        render_spatial_analysis()
    elif active_page == "agents":
        render_agent_orchestrator()
    elif active_page == "settings":
        render_settings()
    else:
        render_chat_interface()


if __name__ == "__main__":
    main()
