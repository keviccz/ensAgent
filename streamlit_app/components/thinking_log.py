"""
Thinking Log and Reasoning Panel components.
Implements progressive disclosure UI for agent reasoning.
"""
from __future__ import annotations

import html
import streamlit as st
from typing import List

from streamlit_app.utils.state import get_state, ReasoningStep
from streamlit_app.styles import get_reasoning_step, get_thinking_animation


REASONING_MODELS = {
    "o1-preview",
    "o1-mini",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
}


def _safe_html_text(text: str) -> str:
    return html.escape(str(text), quote=False).replace("\n", "<br/>")


def _model_supports_reasoning(model_name: str) -> bool:
    """Check if the model supports extended thinking/reasoning."""
    return model_name in REASONING_MODELS


def render_thinking_log() -> None:
    """Render the thinking/reasoning log panel."""
    current_model = get_state("model_name", "gpt-4o")
    supports_reasoning = _model_supports_reasoning(current_model)

    st.markdown(
        """
        <div class="ens-section">
            <div class="ens-section-title">Agent Reasoning</div>
            <p class="ens-section-subtitle">
                Step-by-step reasoning trace from the EnsAgent system.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not supports_reasoning:
        safe_model = _safe_html_text(current_model)
        st.markdown(
            f"""
            <div class="ens-alert">
                <div class="ens-alert-header">
                    <span class="ens-alert-icon">Info</span>
                    <span class="ens-alert-title">Extended Thinking Not Available</span>
                </div>
                <p class="ens-alert-text">
                    Current model (<strong>{safe_model}</strong>) does not support extended thinking traces.
                    Switch to one of these models for detailed reasoning:
                </p>
                <div class="ens-chip-row ens-chip-row-spaced">
                    <span class="ens-chip">claude-opus-4</span>
                    <span class="ens-chip">claude-sonnet-4</span>
                    <span class="ens-chip">o1-preview</span>
                    <span class="ens-chip">o1-mini</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    is_thinking = get_state("is_thinking", False)

    if is_thinking:
        st.markdown(
            f"""
            <div class="ens-thinking-banner">
                {get_thinking_animation()}
                <span class="ens-thinking-text">
                    Agent is thinking...
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    current_reasoning: List[ReasoningStep] = get_state("current_reasoning", [])

    if current_reasoning:
        for step in current_reasoning:
            st.markdown(
                get_reasoning_step(step.step_num, step.title, step.content),
                unsafe_allow_html=True,
            )

    messages = get_state("messages", [])
    assistant_messages = [m for m in messages if m.role == "assistant" and m.reasoning_steps]

    if assistant_messages:
        st.markdown("### Previous Reasoning")

        for msg in reversed(assistant_messages[-5:]):
            with st.expander(
                f"Reasoning ({msg.timestamp.strftime('%H:%M:%S')})",
                expanded=False,
            ):
                for step in msg.reasoning_steps:
                    st.markdown(
                        get_reasoning_step(step.step_num, step.title, step.content),
                        unsafe_allow_html=True,
                    )
    elif not current_reasoning:
        st.markdown(
            """
            <div class="ens-empty-state ens-empty-state-lg">
                <div class="ens-empty-icon">R</div>
                <p class="ens-empty-text">
                    No reasoning traces yet.<br/>
                    Start a conversation to see agent thinking.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_reasoning_panel(steps: List[ReasoningStep], is_live: bool = False) -> None:
    """Render a reasoning panel with given steps."""
    if not steps:
        return

    st.markdown(
        f"""
        <div class="ens-reasoning-panel" data-live="{str(is_live).lower()}">
        """,
        unsafe_allow_html=True,
    )

    for step in steps:
        duration_str = f" ({step.duration_ms}ms)" if step.duration_ms else ""
        safe_title = _safe_html_text(step.title)
        safe_content = _safe_html_text(step.content)

        st.markdown(
            f"""
            <div class="ens-reasoning-step">
                <div class="ens-reasoning-step-header">
                    <span class="ens-reasoning-step-tag">
                        STEP {step.step_num}
                    </span>
                    <span class="ens-reasoning-step-title">
                        {safe_title}
                    </span>
                    <span class="ens-reasoning-step-duration">{duration_str}</span>
                </div>
                <div class="ens-reasoning-step-body">
                    {safe_content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_collapsible_reasoning(steps: List[ReasoningStep], default_open: bool = False) -> None:
    """Render reasoning steps in a collapsible container."""
    if not steps:
        return

    step_count = len(steps)

    with st.expander(f"View reasoning ({step_count} steps)", expanded=default_open):
        for step in steps:
            safe_title = _safe_html_text(step.title)
            safe_content = _safe_html_text(step.content)
            st.markdown(
                f"""
                <div class="ens-reasoning-card">
                    <div class="ens-reasoning-card-title">
                        Step {step.step_num} &middot; {safe_title}
                    </div>
                    <div class="ens-reasoning-card-body">
{safe_content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
