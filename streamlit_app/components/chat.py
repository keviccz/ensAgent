"""
Chat interface component for EnsAgent.
Implements a ChatGPT-style conversational UI with streaming output,
JSON export, and conversation management.
"""
from __future__ import annotations

import html
import json
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict

import streamlit as st

from ensagent_tools import execute_tool, load_config
from streamlit_app.utils.state import (
    get_state,
    set_state,
    add_message,
    start_new_conversation,
    load_conversation,
    ChatMessage,
    ReasoningStep,
)
from streamlit_app.utils.api import EnsAgentAPI, APIError, ENSAGENT_TOOLS


_RESPONSE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ensagent-chat")
_RESPONSE_JOBS: Dict[str, Future[str]] = {}
_RESPONSE_JOBS_LOCK = Lock()


def _safe_html_text(text: str) -> str:
    return html.escape(str(text), quote=False).replace("\n", "<br/>")


def _build_runtime_snapshot() -> Dict[str, Any]:
    """Snapshot runtime settings so background jobs do not depend on session state."""
    model_name = get_state("api_model") or get_state("api_deployment") or get_state("model_name", "gpt-4o")
    return {
        "model_name": model_name,
        "provider": get_state("api_provider") or "",
        "api_key": get_state("api_key") or "",
        "api_endpoint": get_state("api_endpoint") or "",
        "api_version": get_state("api_version") or "",
        "sample_id": get_state("sample_id") or "",
        "data_path": get_state("data_path", "") or "",
        "n_clusters": int(get_state("n_clusters", 7) or 7),
        "temperature": float(get_state("temperature", 0.7)),
        "top_p": float(get_state("top_p", 1.0)),
        "max_tokens": int(get_state("max_tokens", 4096) or 4096),
    }


def _submit_response_job(user_input: str, runtime: Dict[str, Any]) -> str:
    """Submit one background response job and return its id."""
    job_id = uuid.uuid4().hex[:12]
    future = _RESPONSE_EXECUTOR.submit(
        _generate_response_streaming,
        user_input,
        None,
        False,
        runtime,
    )
    with _RESPONSE_JOBS_LOCK:
        _RESPONSE_JOBS[job_id] = future
    return job_id


def _get_response_job(job_id: str | None) -> Future[str] | None:
    if not job_id:
        return None
    with _RESPONSE_JOBS_LOCK:
        return _RESPONSE_JOBS.get(job_id)


def _clear_response_job(job_id: str | None) -> None:
    if not job_id:
        return
    with _RESPONSE_JOBS_LOCK:
        _RESPONSE_JOBS.pop(job_id, None)


def _clear_pending_response_state() -> None:
    set_state("pending_user_input", None)
    set_state("pending_request_conversation_id", None)
    set_state("pending_response_inflight", False)
    set_state("pending_response_job_id", None)


def render_chat_interface() -> None:
    """Render the main chat interface."""
    pending_prompt = get_state("pending_prompt")
    user_input = st.chat_input(
        placeholder="Ask a question or give a command...",
        key="chat_input",
    )
    effective_input = user_input
    if pending_prompt and not effective_input:
        set_state("pending_prompt", None)
        effective_input = pending_prompt

    if effective_input:
        _handle_user_input(effective_input)
        return

    messages_container = st.container()
    with messages_container:
        _render_message_history()


def _render_message_history() -> None:
    """Render the chat message history."""
    messages = get_state("messages", [])
    has_pending = bool(get_state("pending_user_input")) or bool(get_state("pending_response_inflight"))

    if not messages and not has_pending:
        st.markdown('<div class="ens-chat-empty-anchor"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="ens-hero">
                <div class="ens-hero-icon">🧬</div>
                <div class="ens-hero-title">Welcome to EnsAgent</div>
                <p class="ens-hero-subtitle">
                    transcriptomics analysis with multi-agent reasoning, consensus scoring, and BEST domain construction
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container():
            st.markdown('<div class="ens-chip-actions-anchor"></div>', unsafe_allow_html=True)
            chip_labels = [
                "Run end-to-end analysis",
                "Check environment status",
                "Show current config",
            ]
            _, col1, col2, col3, _ = st.columns([0.3, 1, 1, 1, 0.3])
            for col, label in zip([col1, col2, col3], chip_labels):
                with col:
                    if st.button(label, key=f"chip_{label}", type="secondary", use_container_width=True):
                        set_state("pending_prompt", label)
                        st.rerun()
        return

    for msg in messages:
        _render_message(msg)

    current_conv_id = get_state("current_conversation_id")
    pending_conv_id = get_state("pending_request_conversation_id")
    is_pending_here = bool(get_state("pending_response_inflight")) and current_conv_id and pending_conv_id == current_conv_id
    if is_pending_here:
        st.markdown(
            """
            <div class="assistant-message-container">
                <div class="assistant-avatar">🤖</div>
                <div class="assistant-message">EnsAgent is replying<span class="ens-streaming-indicator"> ...</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_message(msg: ChatMessage) -> None:
    """Render a single chat message with proper alignment."""
    if msg.role == "user":
        safe_content = _safe_html_text(msg.content)
        st.markdown(
            f"""
            <div class="user-message-container">
                <div class="user-message">{safe_content}</div>
                <div class="user-avatar">👤</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif msg.role == "assistant":
        safe_content = _safe_html_text(msg.content)
        st.markdown(
            f"""
            <div class="assistant-message-container">
                <div class="assistant-avatar">🤖</div>
                <div class="assistant-message">{safe_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif msg.role == "system":
        st.warning(msg.content)


def _handle_user_input(user_input: str) -> None:
    """Queue a user message and process response in a page-independent phase."""
    if not get_state("current_conversation_id"):
        start_new_conversation()

    stale_job_id = get_state("pending_response_job_id")
    _clear_response_job(stale_job_id)
    add_message("user", user_input)
    set_state("pending_user_input", user_input)
    set_state("pending_request_conversation_id", get_state("current_conversation_id"))
    set_state("pending_response_inflight", False)
    set_state("pending_response_job_id", None)
    st.rerun()


def _generate_response_streaming(
    user_input: str,
    placeholder=None,
    stream: bool = True,
    runtime: Dict[str, Any] | None = None,
) -> str:
    """Generate response with streaming effect."""
    runtime_ctx = runtime or _build_runtime_snapshot()
    model_name = runtime_ctx.get("model_name") or "gpt-4o"
    provider = runtime_ctx.get("provider") or ""
    temperature = float(runtime_ctx.get("temperature", 0.7) or 0.7)
    top_p = float(runtime_ctx.get("top_p", 1.0) or 1.0)
    max_tokens = int(runtime_ctx.get("max_tokens", 4096) or 4096)

    api = EnsAgentAPI(
        model=model_name,
        provider=provider or "",
        api_key=str(runtime_ctx.get("api_key") or ""),
        endpoint=str(runtime_ctx.get("api_endpoint") or ""),
        api_version=str(runtime_ctx.get("api_version") or ""),
    )

    if not api.initialize():
        response = _prepend_local_fallback_notice(
            _get_fallback_response(user_input),
            reason="API not configured in Settings",
        )
        return _stream_text(response, placeholder, stream=stream)

    sample_id = runtime_ctx.get("sample_id") or ""
    data_path = runtime_ctx.get("data_path") or ""
    n_clusters = int(runtime_ctx.get("n_clusters", 7) or 7)

    system_prompt = f"""\
You are EnsAgent, a specialist assistant for spatial transcriptomics ensemble analysis.

## What You Do
You orchestrate a four-stage pipeline that integrates eight spatial clustering methods \
(IRIS, BASS, DR-SC, BayesSpace, SEDR, GraphST, STAGATE, stLearn) to produce consensus \
domain labels from 10X Visium data.

## Pipeline Stages
  A. Tool-Runner  — Run clustering in R/PY/PY2 environments, align labels, produce CSVs.
  B. Scoring      — LLM-driven per-domain evaluation, build score & label matrices.
  C. BEST Builder — Select best label per spot, output BEST_* files + result image.
  D. Annotation   — Multi-agent (VLM + Peer + Critic) domain annotation.

Stages depend on each other: A → B → C → D.

## Current Configuration
- Sample ID : {sample_id or '(not set)'}
- Data Path : {data_path or '(not set)'}
- n_clusters: {n_clusters}
- Temperature: {temperature}
- Top-p      : {top_p}

## Available Tools
check_envs, setup_envs, run_tool_runner, run_scoring, run_best_builder, \
run_annotation, run_end_to_end, show_config, set_config.

## Policies
- Before running, verify config is set. If data_path or sample_id are empty, ask the user.
- Before the first run, check environments.
- Explain what each stage will do before launching it.
- When a tool fails, report the error clearly and suggest a fix.
- When discussing runtime parameters (n_clusters, temperature, top_p), use current config or tool return values only.
- Respond in the same language the user writes in.
- Be concise. Use structured lists."""

    api_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    try:
        max_tool_rounds = 8
        for round_idx in range(max_tool_rounds):
            result = api.chat(
                messages=api_messages,
                tools=ENSAGENT_TOOLS,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            content = result.get("content", "") or ""
            tool_calls = result.get("tool_calls") or []

            if not tool_calls:
                final_content = content or _prepend_local_fallback_notice(
                    _get_fallback_response(user_input),
                    reason="model returned empty content",
                )
                return _stream_text(final_content, placeholder, stream=stream)

            cfg = load_config()
            normalized_tool_calls = []
            for tool_idx, tc in enumerate(tool_calls):
                tc_id = str(tc.get("id") or f"tool_{round_idx}_{tool_idx}")
                name = str(tc.get("name") or "")
                arguments = tc.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {"raw": str(arguments)}

                normalized_tool_calls.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                )

            api_messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": normalized_tool_calls,
                }
            )

            for tc in normalized_tool_calls:
                function = tc["function"]
                name = function["name"]
                try:
                    args = json.loads(function["arguments"]) if function.get("arguments") else {}
                except Exception:
                    args = {}

                out = execute_tool(name, args, cfg)
                tool_response = json.dumps(out, ensure_ascii=False, default=str)
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_response,
                    }
                )
                # set_config may change on-disk pipeline config; reload before next tool.
                if name == "set_config":
                    cfg = load_config()

        loop_content = _prepend_local_fallback_notice(
            _get_fallback_response(user_input),
            reason="tool-call loop reached maximum rounds",
        )
        return _stream_text(loop_content, placeholder, stream=stream)

    except Exception as exc:
        response = _prepend_local_fallback_notice(
            _get_fallback_response(user_input),
            reason=f"request failed ({type(exc).__name__})",
        )
        return _stream_text(response, placeholder, stream=stream)


def _stream_text(text: str, placeholder, *, stream: bool = True) -> str:
    """Stream text in small chunks to reduce layout jitter."""
    if not stream or placeholder is None:
        return text

    displayed = ""

    words = text.split(" ")
    chunk_size = 3
    for i in range(0, len(words), chunk_size):
        displayed = " ".join(words[: i + chunk_size])

        safe_displayed = _safe_html_text(displayed)
        placeholder.markdown(
            f"""
            <div class="assistant-message-container">
                <div class="assistant-avatar">🤖</div>
                <div class="assistant-message">{safe_displayed}<span class="ens-streaming-indicator"> ...</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.015)

    safe_displayed = _safe_html_text(displayed)
    placeholder.markdown(
        f"""
        <div class="assistant-message-container">
            <div class="assistant-avatar">🤖</div>
            <div class="assistant-message">{safe_displayed}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return displayed


def _prepend_local_fallback_notice(text: str, *, reason: str) -> str:
    """Attach an explicit notice when response is local fallback."""
    return f"[Local fallback reply: {reason}]\n\n{text}"


def _get_fallback_response(user_input: str) -> str:
    """Generate a fallback response without API."""
    input_lower = user_input.lower()

    if any(x in input_lower for x in ["run", "execute", "start"]):
        if "end-to-end" in input_lower or "end to end" in input_lower or "pipeline" in input_lower:
            return "I'll run the end-to-end pipeline. Configure Data Path and Sample ID in Settings first, then use the chat quick action: Run end-to-end analysis."
        if "tool" in input_lower or "cluster" in input_lower:
            return "I'll run the Tool Runner for spatial clustering. Make sure your conda environments are set up first."

    if any(x in input_lower for x in ["check", "status", "env"]):
        return "Use the chat quick action: Check environment status. This verifies whether R, PY, and PY2 conda environments are configured."

    if any(x in input_lower for x in ["config", "setting", "show"]):
        return """Your current configuration is available on the Settings page. Key settings include:
- **Data Path**: Location of your spatial transcriptomics data
- **Sample ID**: Unique identifier for your sample
- **Deployment**: The LLM deployment used for agent reasoning"""

    if any(x in input_lower for x in ["help", "what can", "how to"]):
        return """I can help you with:

**Pipeline Operations**
- Run end-to-end analysis
- Execute individual pipeline stages (Tool Runner, Scoring, BEST, Annotation)

**Configuration**
- Set up conda environments
- Configure data paths and sample IDs

**Analysis**
- View spatial clustering results
- Check domain annotations

Just tell me what you'd like to do!"""

    return f"I understand you're asking about: \"{user_input}\"\n\nFor full functionality, configure API credentials in Settings. In the meantime, you can use the chat quick actions to run pipeline operations."


def process_pending_response() -> None:
    """Process queued chat requests regardless of current page."""
    pending_input = get_state("pending_user_input")
    if not pending_input:
        return

    target_conv_id = get_state("pending_request_conversation_id")
    if not target_conv_id:
        _clear_pending_response_state()
        return

    job_id = get_state("pending_response_job_id")
    future = _get_response_job(job_id)
    if not job_id or future is None:
        runtime = _build_runtime_snapshot()
        new_job_id = _submit_response_job(pending_input, runtime)
        set_state("pending_response_job_id", new_job_id)
        set_state("pending_response_inflight", True)
        return

    if not future.done():
        set_state("pending_response_inflight", True)
        return

    original_conv_id = get_state("current_conversation_id")
    switched_context = False
    try:
        if original_conv_id != target_conv_id:
            switched_context = load_conversation(target_conv_id)
            if not switched_context:
                raise RuntimeError(f"Conversation not found: {target_conv_id}")

        full_response = future.result()
        add_message("assistant", full_response)
    except Exception as exc:
        add_message("system", f"Error: {exc}")
    finally:
        _clear_response_job(job_id)
        _clear_pending_response_state()
        if switched_context and original_conv_id and original_conv_id != get_state("current_conversation_id"):
            load_conversation(original_conv_id)
