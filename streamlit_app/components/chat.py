"""
Chat interface component for EnsAgent.
Implements a ChatGPT-style conversational UI with streaming output,
JSON export, and conversation management.
"""
from __future__ import annotations

import html
import importlib
import json
import time
import uuid
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock
from typing import Any, Dict

import streamlit as st

from ensagent_tools import execute_tool, load_config
from streamlit_app.utils.state import (
    add_pipeline_log,
    get_state,
    set_state,
    add_message,
    start_new_conversation,
    load_conversation,
    ChatMessage,
    ReasoningStep,
    reset_all_agents,
    update_agent_status,
)
from streamlit_app.utils.api import EnsAgentAPI, ENSAGENT_TOOLS


_RESPONSE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ensagent-chat")


class ResponseInterrupted(Exception):
    """Raised when a queued response is interrupted by a newer user request."""


@dataclass
class ResponseJob:
    future: Future[str]
    cancel_event: Event | None
    request_id: str
    progress_queue: Queue[Dict[str, Any]] | None = None


_RESPONSE_JOBS: Dict[str, ResponseJob] = {}
_RESPONSE_JOBS_LOCK = Lock()
_EXECUTE_TOOL_REBIND_LOCK = Lock()
_EXECUTE_TOOL_REBIND_ATTEMPTED = False

_STAGE_ACTION_TOKENS = (
    "run",
    "execute",
    "start",
    "launch",
    "运行",
    "执行",
    "开始",
    "启动",
)

_PIPELINE_KEYWORDS = (
    "end-to-end",
    "end to end",
    "full pipeline",
    "run all",
    "all stages",
    "全流程",
    "全部阶段",
    "一键",
)

_STAGE_A_KEYWORDS = ("stagea", "stage a", "阶段a", "tool-runner", "tool runner")
_STAGE_B_KEYWORDS = ("stageb", "stage b", "阶段b", "scoring")
_STAGE_C_KEYWORDS = ("stagec", "stage c", "阶段c", "best", "best builder")
_STAGE_D_KEYWORDS = ("staged", "stage d", "阶段d", "annotation")

_TOOL_TO_STAGE = {
    "run_tool_runner": "tool_runner",
    "run_scoring": "scoring",
    "run_best_builder": "best",
    "run_annotation": "annotation",
}

_TOOL_TO_AGENT = {
    "run_tool_runner": "Tool Runner",
    "run_scoring": "Scoring Agent",
    "run_best_builder": "BEST Builder",
    "run_annotation": "Annotation Agent",
}


def _safe_html_text(text: str) -> str:
    return html.escape(str(text), quote=False).replace("\n", "<br/>")


def _execute_tool_compat(
    name: str,
    args: Dict[str, Any],
    cfg: Any,
    *,
    progress_callback=None,
    cancel_check=None,
) -> Dict[str, Any]:
    """
    Execute a tool with forward/backward compatibility.

    New runtime expects execute_tool(..., progress_callback=..., cancel_check=...),
    while older runtime only accepts execute_tool(name, args, cfg).
    """
    def _call_execute_tool(with_kwargs: bool) -> Dict[str, Any]:
        if with_kwargs:
            return execute_tool(
                name,
                args,
                cfg,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
        return execute_tool(name, args, cfg)

    try:
        return _call_execute_tool(with_kwargs=True)
    except TypeError as exc:
        err = str(exc)
        legacy_sig_mismatch = (
            "unexpected keyword argument" in err
            and ("progress_callback" in err or "cancel_check" in err)
        )
        if not legacy_sig_mismatch:
            raise

        # Some long-lived sessions can keep a stale execute_tool binding after upgrades.
        # Try one in-process rebind to the latest ensagent_tools export, then retry.
        global _EXECUTE_TOOL_REBIND_ATTEMPTED
        should_attempt_rebind = False
        with _EXECUTE_TOOL_REBIND_LOCK:
            if not _EXECUTE_TOOL_REBIND_ATTEMPTED:
                _EXECUTE_TOOL_REBIND_ATTEMPTED = True
                should_attempt_rebind = True

        if should_attempt_rebind:
            tool_module = str(getattr(execute_tool, "__module__", "") or "")
            if tool_module.startswith("ensagent_tools"):
                try:
                    import ensagent_tools as _ensagent_tools
                    import ensagent_tools.registry as _ensagent_registry

                    importlib.reload(_ensagent_registry)
                    importlib.reload(_ensagent_tools)
                    globals()["execute_tool"] = getattr(_ensagent_tools, "execute_tool", execute_tool)
                    return _call_execute_tool(with_kwargs=True)
                except Exception:
                    pass

        return _call_execute_tool(with_kwargs=False)


def _normalize_log_level(level: str | None) -> str:
    normalized = str(level or "info").strip().lower()
    if normalized in {"info", "warning", "error", "success"}:
        return normalized
    return "info"


def _tool_stage_key(tool_name: str, stage_name: str) -> str | None:
    if tool_name in _TOOL_TO_STAGE:
        return _TOOL_TO_STAGE[tool_name]
    normalized_stage = str(stage_name or "").strip().lower()
    if "tool" in normalized_stage:
        return "tool_runner"
    if "scor" in normalized_stage:
        return "scoring"
    if "best" in normalized_stage:
        return "best"
    if "annot" in normalized_stage:
        return "annotation"
    return None


def _tool_agent_name(tool_name: str, stage_name: str) -> str | None:
    if tool_name in _TOOL_TO_AGENT:
        return _TOOL_TO_AGENT[tool_name]
    stage_key = _tool_stage_key(tool_name, stage_name)
    if stage_key == "tool_runner":
        return "Tool Runner"
    if stage_key == "scoring":
        return "Scoring Agent"
    if stage_key == "best":
        return "BEST Builder"
    if stage_key == "annotation":
        return "Annotation Agent"
    return None


def _set_agent_progress(agent_name: str, progress: float) -> None:
    agents = get_state("agents", []) or []
    try:
        safe_progress = max(0.0, min(1.0, float(progress)))
    except Exception:
        return
    for agent in agents:
        if getattr(agent, "name", "") == agent_name:
            agent.progress = safe_progress
            return


def _append_pending_live_log(line: str, *, max_lines: int = 16) -> None:
    current = list(get_state("pending_live_log_lines", []) or [])
    current.append(str(line))
    if len(current) > max_lines:
        current = current[-max_lines:]
    set_state("pending_live_log_lines", current)


def _apply_progress_event(event: Dict[str, Any]) -> None:
    kind = str(event.get("kind") or "tool_log")
    tool_name = str(event.get("tool") or "")
    stage_name = str(event.get("stage") or "")
    message = str(event.get("message") or "").strip()
    level = _normalize_log_level(str(event.get("level") or "info"))
    progress_raw = event.get("progress")

    stage_key = _tool_stage_key(tool_name, stage_name)
    agent_name = _tool_agent_name(tool_name, stage_name)

    if stage_key:
        set_state("pipeline_stage", stage_key)
    if message:
        log_prefix = f"[{tool_name}] " if tool_name else ""
        add_pipeline_log(f"{log_prefix}{message}", level=level)
        _append_pending_live_log(message)

    progress_val: float | None = None
    if progress_raw is not None:
        try:
            progress_val = max(0.0, min(1.0, float(progress_raw)))
        except Exception:
            progress_val = None

    if kind == "tool_start":
        if stage_key:
            set_state("pipeline_progress", 0.0)
        if agent_name:
            update_agent_status(agent_name, "active", message or "Running")
            _set_agent_progress(agent_name, 0.0)
        return

    if kind == "tool_progress":
        if progress_val is not None:
            if stage_key:
                set_state("pipeline_progress", progress_val)
            if agent_name:
                update_agent_status(agent_name, "active", message or "Running")
                _set_agent_progress(agent_name, progress_val)
        return

    if kind == "tool_end":
        if stage_key:
            set_state("pipeline_progress", 1.0)
        if agent_name:
            update_agent_status(agent_name, "completed", message or "Completed")
            _set_agent_progress(agent_name, 1.0)
        return

    if kind == "tool_error":
        if agent_name:
            update_agent_status(agent_name, "error", message or "Failed")
        return

    if kind == "tool_interrupt":
        if agent_name:
            update_agent_status(agent_name, "idle", message or "Interrupted")
            _set_agent_progress(agent_name, 0.0)
        return


def _drain_progress_queue(job: ResponseJob | None) -> None:
    if job is None or job.progress_queue is None:
        return
    while True:
        try:
            event = job.progress_queue.get_nowait()
        except Empty:
            break
        if isinstance(event, dict):
            _apply_progress_event(event)


def _hidden_quick_actions_set() -> set[str]:
    raw = get_state("_quick_actions_hidden_conversations", []) or []
    if isinstance(raw, (list, tuple, set)):
        return {str(item) for item in raw if item}
    return set()


def _hide_quick_actions_for_current_conversation() -> None:
    conv_id = get_state("current_conversation_id")
    if not conv_id:
        return
    hidden = _hidden_quick_actions_set()
    if conv_id in hidden:
        return
    hidden.add(conv_id)
    set_state("_quick_actions_hidden_conversations", sorted(hidden))


def _should_show_quick_actions(messages: list[ChatMessage], has_pending: bool) -> bool:
    if messages or has_pending:
        return False
    conv_id = get_state("current_conversation_id")
    if not conv_id:
        return True
    return conv_id not in _hidden_quick_actions_set()


def _chat_ui_has_pending_activity() -> bool:
    """Return True while chat UI has queued or in-flight work."""
    return bool(
        get_state("pending_user_input")
        or get_state("pending_response_inflight")
        or get_state("pending_prompt")
    )


def _queue_quick_action_prompt(prompt: str) -> None:
    """Queue a quick prompt and hide quick-action chips for that conversation."""
    if not get_state("current_conversation_id"):
        start_new_conversation()
    _hide_quick_actions_for_current_conversation()
    set_state("pending_prompt", prompt)


def consume_pending_action_prompt() -> None:
    """Convert dashboard pending_action into a queued chat prompt."""
    action = str(get_state("pending_action") or "").strip()
    if not action:
        return
    mapping = {
        "run_end_to_end": "Run end-to-end analysis",
        "check_envs": "Check environment status",
        "show_config": "Show current config",
    }
    set_state("pending_action", None)
    prompt = mapping.get(action)
    if not prompt:
        return
    set_state("active_page", "chat")
    set_state("pending_prompt", prompt)


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
        "csv_path": get_state("csv_path", "") or "",
        "n_clusters": int(get_state("n_clusters", 7) or 7),
        "temperature": float(get_state("temperature", 0.7)),
        "top_p": float(get_state("top_p", 1.0)),
        "max_tokens": int(get_state("max_tokens", 4096) or 4096),
    }


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _prefers_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _effective_runtime_value(runtime: Dict[str, Any], cfg: Any, key: str) -> str:
    runtime_val = str(runtime.get(key) or "").strip()
    if runtime_val:
        return runtime_val
    cfg_val = str(getattr(cfg, key, "") or "").strip()
    return cfg_val


def _resolve_scoring_input_dir(runtime: Dict[str, Any], cfg: Any) -> str:
    runtime_csv = str(runtime.get("csv_path") or "").strip()
    if runtime_csv:
        return runtime_csv
    cfg_csv = str(getattr(cfg, "csv_path", "") or "").strip()
    if cfg_csv:
        return cfg_csv
    return str(cfg.resolved_scoring_input_dir())


def _detect_direct_tool_intent(user_input: str) -> str | None:
    text = (user_input or "").strip().lower()
    if not text:
        return None

    has_stage_action = _contains_any(text, _STAGE_ACTION_TOKENS)
    if has_stage_action and _contains_any(text, _PIPELINE_KEYWORDS):
        return "run_end_to_end"
    if has_stage_action and _contains_any(text, _STAGE_A_KEYWORDS):
        return "run_tool_runner"
    if has_stage_action and _contains_any(text, _STAGE_B_KEYWORDS):
        return "run_scoring"
    if has_stage_action and _contains_any(text, _STAGE_C_KEYWORDS):
        return "run_best_builder"
    if has_stage_action and _contains_any(text, _STAGE_D_KEYWORDS):
        return "run_annotation"
    return None


def _validate_direct_tool_prereqs(tool_name: str, runtime: Dict[str, Any], cfg: Any, *, zh: bool) -> str | None:
    data_path = _effective_runtime_value(runtime, cfg, "data_path")
    sample_id = _effective_runtime_value(runtime, cfg, "sample_id")
    scoring_input = _resolve_scoring_input_dir(runtime, cfg)

    if tool_name in {"run_tool_runner", "run_end_to_end"}:
        missing: list[str] = []
        if not data_path:
            missing.append("data_path")
        if not sample_id:
            missing.append("sample_id")
        if missing:
            if zh:
                return (
                    f"无法执行 `{tool_name}`，缺少必填配置：{', '.join(missing)}。\n"
                    "请先在 Settings 中补齐后重试。"
                )
            return (
                f"Cannot execute `{tool_name}` because required settings are missing: {', '.join(missing)}.\n"
                "Please set them in Settings and try again."
            )

    if tool_name == "run_scoring":
        if not scoring_input:
            if zh:
                return "无法执行 `run_scoring`，未找到 Scoring CSV 输入目录（csv_path）。"
            return "Cannot execute `run_scoring` because no scoring CSV input directory (`csv_path`) is configured."
        if not Path(scoring_input).exists():
            if zh:
                return (
                    "无法执行 `run_scoring`，CSV 输入目录不存在：\n"
                    f"{scoring_input}\n"
                    "请在 Settings 中设置可用的 Scoring CSV Path。"
                )
            return (
                "Cannot execute `run_scoring` because the scoring CSV input directory does not exist:\n"
                f"{scoring_input}\n"
                "Please set a valid Scoring CSV Path in Settings."
            )

    if tool_name in {"run_best_builder", "run_annotation"} and not sample_id:
        if zh:
            return f"无法执行 `{tool_name}`，缺少 `sample_id`。请先在 Settings 中设置 Sample ID。"
        return f"Cannot execute `{tool_name}` because `sample_id` is missing. Set Sample ID in Settings first."

    return None


def _build_direct_tool_args(tool_name: str, runtime: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    data_path = _effective_runtime_value(runtime, cfg, "data_path")
    sample_id = _effective_runtime_value(runtime, cfg, "sample_id")
    scoring_input = _resolve_scoring_input_dir(runtime, cfg)

    if tool_name == "run_tool_runner":
        return {
            "data_path": data_path,
            "sample_id": sample_id,
            "n_clusters": int(runtime.get("n_clusters", getattr(cfg, "n_clusters", 7)) or 7),
        }
    if tool_name == "run_scoring":
        return {
            "input_dir": scoring_input,
            "temperature": float(runtime.get("temperature", getattr(cfg, "temperature", 0.7)) or 0.7),
            "top_p": float(runtime.get("top_p", getattr(cfg, "top_p", 1.0)) or 1.0),
        }
    if tool_name == "run_best_builder":
        return {"sample_id": sample_id}
    if tool_name == "run_annotation":
        return {"sample_id": sample_id}
    return {}


def _format_direct_tool_result(tool_name: str, tool_result: Any, *, zh: bool) -> str:
    result = tool_result if isinstance(tool_result, dict) else {"ok": False, "raw": str(tool_result)}
    ok = bool(result.get("ok"))
    if zh:
        header = f"已执行 `{tool_name}`。" if ok else f"`{tool_name}` 执行失败。"
        label = "结果"
    else:
        header = f"Executed `{tool_name}`." if ok else f"`{tool_name}` failed."
        label = "Result"
    payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    return f"{header}\n\n{label}:\n{payload}"


def _submit_response_job(user_input: str, runtime: Dict[str, Any], request_id: str) -> str:
    """Submit one background response job and return its id."""
    job_id = uuid.uuid4().hex[:12]
    cancel_event = Event()
    progress_queue: Queue[Dict[str, Any]] = Queue()

    def _emit_progress(event: Dict[str, Any]) -> None:
        try:
            progress_queue.put_nowait(dict(event or {}))
        except Exception:
            return

    future = _RESPONSE_EXECUTOR.submit(
        _generate_response_streaming,
        user_input,
        None,
        False,
        runtime,
        cancel_event,
        _emit_progress,
    )
    with _RESPONSE_JOBS_LOCK:
        _RESPONSE_JOBS[job_id] = ResponseJob(
            future=future,
            cancel_event=cancel_event,
            request_id=request_id,
            progress_queue=progress_queue,
        )
    return job_id


def _get_response_job(job_id: str | None) -> ResponseJob | None:
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
    set_state("pending_request_id", None)
    set_state("pending_response_inflight", False)
    set_state("pending_response_job_id", None)
    set_state("pending_live_log_lines", [])


def _append_system_message_to_conversation(conv_id: str, text: str) -> None:
    """Append a system message into a target conversation without losing current context."""
    if not conv_id:
        return
    original_conv_id = get_state("current_conversation_id")
    switched_context = False
    try:
        if original_conv_id != conv_id:
            switched_context = load_conversation(conv_id)
            if not switched_context:
                return
        add_message("system", text)
    finally:
        if switched_context and original_conv_id and original_conv_id != get_state("current_conversation_id"):
            load_conversation(original_conv_id)


def _interrupt_active_response() -> None:
    """Interrupt the current in-flight response and mark it in chat history."""
    old_input = get_state("pending_user_input")
    old_conv_id = get_state("pending_request_conversation_id")
    old_job_id = get_state("pending_response_job_id")
    old_job = _get_response_job(old_job_id)

    if old_job is not None:
        if old_job.cancel_event is not None:
            old_job.cancel_event.set()
        old_job.future.cancel()
        _clear_response_job(old_job_id)

    if old_input and old_conv_id:
        _append_system_message_to_conversation(old_conv_id, "[已中断上一条回复]")

    _clear_pending_response_state()


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
    has_pending = _chat_ui_has_pending_activity()

    if _should_show_quick_actions(messages, has_pending):
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
                        _queue_quick_action_prompt(label)
                        st.rerun()
        return

    for msg in messages:
        _render_message(msg)

    current_conv_id = get_state("current_conversation_id")
    pending_conv_id = get_state("pending_request_conversation_id")
    is_pending_here = bool(get_state("pending_response_inflight")) and current_conv_id and pending_conv_id == current_conv_id
    if is_pending_here:
        live_logs = list(get_state("pending_live_log_lines", []) or [])
        live_logs_html = ""
        if live_logs:
            body = "".join(
                f"<div class='ens-live-log-line'>{_safe_html_text(line)}</div>"
                for line in live_logs[-8:]
            )
            live_logs_html = f"<div class='ens-live-log-box'>{body}</div>"
        st.markdown(
            """
            <div class="assistant-message-container">
                <div class="assistant-avatar">🤖</div>
                <div class="assistant-message">"""
            + live_logs_html
            + """<span class="ens-typing-caret" aria-label="Assistant typing indicator"></span></div>
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
    _hide_quick_actions_for_current_conversation()

    has_pending = bool(get_state("pending_user_input")) or bool(get_state("pending_response_inflight"))
    if has_pending:
        _interrupt_active_response()

    add_message("user", user_input)
    reset_all_agents()
    set_state("pipeline_stage", None)
    set_state("pipeline_progress", 0.0)
    set_state("pending_live_log_lines", [])
    set_state("pending_user_input", user_input)
    set_state("pending_request_conversation_id", get_state("current_conversation_id"))
    set_state("pending_request_id", uuid.uuid4().hex[:12])
    set_state("pending_response_inflight", False)
    set_state("pending_response_job_id", None)
    st.rerun()


def _generate_response_streaming(
    user_input: str,
    placeholder=None,
    stream: bool = True,
    runtime: Dict[str, Any] | None = None,
    cancel_event: Event | None = None,
    progress_callback=None,
) -> str:
    """Generate response with streaming effect."""
    if cancel_event is not None and cancel_event.is_set():
        raise ResponseInterrupted("response interrupted before start")

    runtime_ctx = runtime or _build_runtime_snapshot()
    model_name = runtime_ctx.get("model_name") or "gpt-4o"
    provider = runtime_ctx.get("provider") or ""
    temperature = float(runtime_ctx.get("temperature", 0.7) or 0.7)
    top_p = float(runtime_ctx.get("top_p", 1.0) or 1.0)
    max_tokens = int(runtime_ctx.get("max_tokens", 4096) or 4096)
    cfg = load_config()
    prefers_zh = _prefers_chinese(user_input)
    cancel_check = (lambda: bool(cancel_event is not None and cancel_event.is_set()))

    direct_tool_name = _detect_direct_tool_intent(user_input)
    if direct_tool_name:
        prereq_issue = _validate_direct_tool_prereqs(direct_tool_name, runtime_ctx, cfg, zh=prefers_zh)
        if prereq_issue:
            if cancel_event is not None and cancel_event.is_set():
                raise ResponseInterrupted("response interrupted by newer input")
            return _stream_text(prereq_issue, placeholder, stream=stream)

        direct_args = _build_direct_tool_args(direct_tool_name, runtime_ctx, cfg)
        direct_result = _execute_tool_compat(
            direct_tool_name,
            direct_args,
            cfg,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        direct_text = _format_direct_tool_result(direct_tool_name, direct_result, zh=prefers_zh)
        if cancel_event is not None and cancel_event.is_set():
            raise ResponseInterrupted("response interrupted by newer input")
        return _stream_text(direct_text, placeholder, stream=stream)

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
    csv_path = runtime_ctx.get("csv_path") or ""
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
- CSV Path  : {csv_path or '(not set)'}
- n_clusters: {n_clusters}
- Temperature: {temperature}
- Top-p      : {top_p}

## Available Tools
check_envs, setup_envs, run_tool_runner, run_scoring, run_best_builder, \
run_annotation, run_end_to_end, show_config, set_config.

## Policies
- If the user explicitly asks to execute a stage (for example: "Run Stage B", "执行StageB"), call the corresponding tool immediately; do not only restate the request.
- Before running Tool-runner, verify data_path and sample_id are set.
- Before running Scoring without Tool-runner, verify csv_path (or scoring/input) is set and exists.
- For explicit execution requests, keep explanation to one brief sentence before tool execution.
- Before the first run, check environments.
- When a tool fails, report the error clearly and suggest a fix.
- If required config is missing, state the missing keys and tell the user exactly what to set in Settings.
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
            if cancel_event is not None and cancel_event.is_set():
                raise ResponseInterrupted("response interrupted by newer input")

            result = api.chat(
                messages=api_messages,
                tools=ENSAGENT_TOOLS,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise ResponseInterrupted("response interrupted by newer input")

            content = result.get("content", "") or ""
            tool_calls = result.get("tool_calls") or []

            if not tool_calls:
                final_content = content or _prepend_local_fallback_notice(
                    _get_fallback_response(user_input),
                    reason="model returned empty content",
                )
                if cancel_event is not None and cancel_event.is_set():
                    raise ResponseInterrupted("response interrupted by newer input")
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
                if cancel_event is not None and cancel_event.is_set():
                    raise ResponseInterrupted("response interrupted by newer input")
                function = tc["function"]
                name = function["name"]
                try:
                    args = json.loads(function["arguments"]) if function.get("arguments") else {}
                except Exception:
                    args = {}

                if progress_callback is not None:
                    try:
                        progress_callback(
                            {
                                "kind": "tool_log",
                                "tool": name,
                                "stage": _TOOL_TO_STAGE.get(name, "orchestrator"),
                                "level": "info",
                                "message": f"Executing tool: {name}",
                            }
                        )
                    except Exception:
                        pass

                out = _execute_tool_compat(
                    name,
                    args,
                    cfg,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
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
        if cancel_event is not None and cancel_event.is_set():
            raise ResponseInterrupted("response interrupted by newer input")
        return _stream_text(loop_content, placeholder, stream=stream)

    except Exception as exc:
        if isinstance(exc, ResponseInterrupted):
            raise
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
                <div class="assistant-message">{safe_displayed}<span class="ens-typing-caret" aria-hidden="true"></span></div>
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


def _mark_response_completion() -> None:
    """Mark one completed response cycle so main loop can force a final refresh."""
    tick = int(get_state("_chat_completion_tick", 0) or 0)
    set_state("_chat_completion_tick", tick + 1)


def process_pending_response() -> None:
    """Process queued chat requests regardless of current page."""
    pending_input = get_state("pending_user_input")
    if not pending_input:
        return

    target_conv_id = get_state("pending_request_conversation_id")
    pending_request_id = get_state("pending_request_id")
    if not target_conv_id:
        _clear_pending_response_state()
        return

    job_id = get_state("pending_response_job_id")
    job = _get_response_job(job_id)
    if not job_id or job is None:
        if not pending_request_id:
            pending_request_id = uuid.uuid4().hex[:12]
            set_state("pending_request_id", pending_request_id)
        runtime = _build_runtime_snapshot()
        new_job_id = _submit_response_job(pending_input, runtime, pending_request_id)
        set_state("pending_response_job_id", new_job_id)
        set_state("pending_response_inflight", True)
        return

    _drain_progress_queue(job)

    if not job.future.done():
        set_state("pending_response_inflight", True)
        return

    latest_request_id = get_state("pending_request_id")
    if latest_request_id and job.request_id != latest_request_id:
        _clear_response_job(job_id)
        set_state("pending_response_inflight", False)
        return

    original_conv_id = get_state("current_conversation_id")
    switched_context = False
    try:
        _drain_progress_queue(job)
        if original_conv_id != target_conv_id:
            switched_context = load_conversation(target_conv_id)
            if not switched_context:
                raise RuntimeError(f"Conversation not found: {target_conv_id}")

        full_response = job.future.result()
        add_message("assistant", full_response)
        _mark_response_completion()
    except (ResponseInterrupted, CancelledError):
        pass
    except Exception as exc:
        add_message("system", f"Error: {exc}")
        _mark_response_completion()
    finally:
        _clear_response_job(job_id)
        _clear_pending_response_state()
        if switched_context and original_conv_id and original_conv_id != get_state("current_conversation_id"):
            load_conversation(original_conv_id)
