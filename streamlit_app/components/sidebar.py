"""
Sidebar component for EnsAgent.
Navigation + conversation history panel.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from streamlit_app.utils.config_bridge import load_pipeline_fields
from streamlit_app.utils.state import (
    get_state,
    set_state,
    get_conversations,
    start_new_conversation,
    load_conversation,
    delete_conversation,
    export_conversation_json,
)


# ── Config persistence helpers (local UI prefs only) ────────────────────

def _get_config_path() -> Path:
    """Get path to local config file."""
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / ".streamlit_config.json"


def _load_saved_config() -> dict:
    """Load saved local UI preferences from local file."""
    config_path = _get_config_path()
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_config(**kwargs) -> None:
    """Save local UI preferences to local file."""
    config_path = _get_config_path()
    try:
        existing = _load_saved_config()
        existing.update(kwargs)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass


def _init_from_saved_config() -> None:
    """Initialize session state from local UI preferences (only on first load)."""
    if get_state("_config_loaded"):
        return

    saved = _load_saved_config()
    if saved:
        for key in [
            "temperature",
            "top_p",
        ]:
            val = saved.get(key)
            if val is not None:
                set_state(key, val)

    set_state("_config_loaded", True)


def _init_from_pipeline_config() -> None:
    """Initialize session state from pipeline_config.yaml (only on first load)."""
    if get_state("_pipeline_config_loaded"):
        return

    loaded = load_pipeline_fields()
    for key in [
        "data_path",
        "sample_id",
        "n_clusters",
        "temperature",
        "top_p",
        "api_provider",
        "api_key",
        "api_endpoint",
        "api_version",
        "api_model",
        "api_deployment",
    ]:
        val = loaded.get(key, "")
        if key in {"n_clusters", "temperature", "top_p"}:
            if val is not None and val != "":
                set_state(key, val)
            continue
        if val:
            set_state(key, val)

    set_state("_pipeline_config_loaded", True)


# ── Navigation definition ───────────────────────────────────────────────

NAV_ITEMS = [
    ("chat", "Chat"),
    ("spatial", "Analysis"),
    ("agents", "Agents"),
    ("settings", "Settings"),
]


# ── Main render ─────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Render the sidebar with navigation and conversation history."""
    _init_from_pipeline_config()
    _init_from_saved_config()

    with st.sidebar:
        # Brand
        st.markdown(
            """
            <div class="ens-sidebar-brand">
                <div class="ens-sidebar-icon">🧬</div>
                <div>
                    <div class="ens-sidebar-title">EnsAgent</div>
                    <div class="ens-sidebar-subtitle">Spatial Transcriptomics</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="ens-spacer-sm"></div>', unsafe_allow_html=True)

        # Navigation
        active_page = get_state("active_page", "chat")

        for page_id, label in NAV_ITEMS:
            is_active = active_page == page_id
            if st.button(
                label,
                key=f"nav_{page_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                set_state("active_page", page_id)
                st.rerun()

        st.markdown('<div class="ens-spacer-sm"></div>', unsafe_allow_html=True)
        st.markdown('<div class="ens-sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="ens-history-anchor"></div>', unsafe_allow_html=True)
        _render_conversation_history()


def _render_conversation_history() -> None:
    """Render compact conversation history in the sidebar."""
    with st.container():
        st.markdown('<div class="ens-new-chat-anchor"></div>', unsafe_allow_html=True)
        if st.button("＋ New Chat", key="new_chat_btn", use_container_width=True, type="secondary"):
            set_state("active_page", "chat")
            start_new_conversation()
            st.rerun()

    conversations = get_conversations()
    if not conversations:
        st.markdown(
            '<p class="ens-history-empty">No conversations yet</p>',
            unsafe_allow_html=True,
        )
        return

    current_id = get_state("current_conversation_id")

    st.markdown(
        '<div class="ens-history-label">Recent</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="ens-history-list-anchor"></div>', unsafe_allow_html=True)
        for conv in conversations[:20]:
            preview = conv.get_preview(36)
            is_active = conv.id == current_id

            col_btn, col_menu = st.columns([7.6, 1.0], vertical_alignment="center")
            with col_btn:
                if st.button(
                    preview,
                    key=f"hist_{conv.id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    _open_conversation_from_history(conv.id)
                    st.rerun()
            with col_menu:
                # Keep label empty so only Streamlit's right-side chevron is shown.
                with st.popover("", use_container_width=True):
                    st.download_button(
                        "Export JSON",
                        data=export_conversation_json(conv),
                        file_name=f"conversation_{conv.id}.json",
                        mime="application/json",
                        use_container_width=True,
                        key=f"export_{conv.id}",
                    )
                    if st.button(
                        "Delete",
                        key=f"del_{conv.id}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        delete_conversation(conv.id)
                        st.rerun()


def _open_conversation_from_history(conv_id: str) -> None:
    """Load a history conversation and route UI back to Chat page."""
    set_state("active_page", "chat")
    load_conversation(conv_id)
