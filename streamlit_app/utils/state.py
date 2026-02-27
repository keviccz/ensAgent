"""
Centralized session state management for EnsAgent.
Provides a clean interface for multi-step agentic workflows.
Includes persistent disk cache for chat history.
"""
from __future__ import annotations

import json as _json
import streamlit as st
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


_HISTORY_DIR = Path(__file__).resolve().parent.parent.parent / "chat_history"
_LEGACY_HISTORY_DIR = Path(__file__).resolve().parent.parent.parent / ".chat_history"


@dataclass
class AgentStatus:
    """Status of a sub-agent in the pipeline."""
    name: str
    icon: str
    status: str = "idle"  # idle, active, completed, error
    description: str = ""
    last_update: Optional[datetime] = None
    progress: float = 0.0


@dataclass
class ReasoningStep:
    """A single step in the agent's reasoning process."""
    step_num: int
    title: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[int] = None


@dataclass
class ChatMessage:
    """A chat message in the conversation."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    is_thinking: bool = False


@dataclass
class Conversation:
    """A conversation containing multiple messages."""
    id: str
    title: str
    messages: List[ChatMessage]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def get_preview(self, max_len: int = 40) -> str:
        """Get a preview of the conversation."""
        if not self.messages:
            return "New conversation"
        # Get first user message as preview
        for msg in self.messages:
            if msg.role == "user":
                content = msg.content[:max_len]
                return content + "..." if len(msg.content) > max_len else content
        return "New conversation"


# Default agent configurations
DEFAULT_AGENTS = [
    AgentStatus("Data Processor", "DP", "idle", "Handles data ingestion and preprocessing"),
    AgentStatus("Tool Runner", "TR", "idle", "Orchestrates spatial clustering methods"),
    AgentStatus("Scoring Agent", "SA", "idle", "Evaluates and aggregates domain scores"),
    AgentStatus("BEST Builder", "BB", "idle", "Selects optimal domain labels"),
    AgentStatus("Annotation Agent", "AA", "idle", "Multi-agent domain annotation"),
    AgentStatus("Critic Agent", "CR", "idle", "Validates and refines annotations"),
]


def init_session_state() -> None:
    """Initialize all session state variables with defaults."""
    defaults = {
        # Chat state
        "messages": [],  # Current conversation messages
        "conversations": [],  # List of Conversation objects
        "current_conversation_id": None,  # ID of current conversation
        "is_thinking": False,
        "current_reasoning": [],
        
        # Agent state
        "agents": [AgentStatus(a.name, a.icon, a.status, a.description) for a in DEFAULT_AGENTS],
        "active_agent": None,
        
        # Pipeline state
        "pipeline_stage": None,  # None, "tool_runner", "scoring", "best", "annotation"
        "pipeline_progress": 0.0,
        "pipeline_logs": [],
        
        # Data state
        "uploaded_files": {},
        "data_path": None,
        "sample_id": None,
        "output_dir": None,
        
        # API provider state
        "api_provider": None,  # "azure", "openai", "anthropic" or None (auto-detect)
        
        # Results state
        "spatial_data": None,
        "scores_matrix": None,
        "labels_matrix": None,
        "annotations": None,
        "visualization_data": None,
        
        # Config state
        "api_key": "",
        "api_endpoint": "",
        "api_version": None,
        "api_model": None,
        "api_deployment": None,
        "model_name": None,
        "temperature": 0.7,
        "top_p": 1.0,
        "n_clusters": 7,
        "max_tokens": 4096,
        "show_reasoning": True,
        
        # UI state
        "sidebar_collapsed": False,
        "active_page": "chat",
        "active_tab": "Overview",
        "pending_prompt": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.get("active_page") == "home":
        st.session_state["active_page"] = "chat"

    if "agents" in st.session_state:
        icon_map = {agent.name: agent.icon for agent in DEFAULT_AGENTS}
        for agent in st.session_state["agents"]:
            if agent.name in icon_map:
                agent.icon = icon_map[agent.name]

    # Hydrate conversations from disk on first load
    if not st.session_state.get("_history_loaded"):
        disk_convs = load_conversations_from_disk()
        existing_ids = {c.id for c in st.session_state.get("conversations", [])}
        for c in disk_convs:
            if c.id not in existing_ids:
                st.session_state.conversations.append(c)
        st.session_state["_history_loaded"] = True


def get_state(key: str, default: Any = None) -> Any:
    """Get a value from session state."""
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """Set a value in session state."""
    st.session_state[key] = value


def add_message(role: str, content: str, reasoning_steps: Optional[List[ReasoningStep]] = None) -> None:
    """Add a message to the current conversation and persist to disk."""
    msg = ChatMessage(
        role=role,
        content=content,
        reasoning_steps=reasoning_steps or [],
    )
    st.session_state.messages.append(msg)

    conv_id = st.session_state.get("current_conversation_id")
    if conv_id:
        for conv in st.session_state.conversations:
            if conv.id == conv_id:
                conv.messages = st.session_state.messages.copy()
                conv.updated_at = datetime.now()
                if not conv.title or conv.title == "New conversation":
                    for m in conv.messages:
                        if m.role == "user":
                            conv.title = m.content[:50] + ("..." if len(m.content) > 50 else "")
                            break
                persist_conversation(conv)
                break


def start_new_conversation() -> str:
    """Start a new conversation and return its ID."""
    if st.session_state.messages and st.session_state.current_conversation_id:
        _save_current_conversation(touch_updated_at=False)

    conv_id = str(uuid.uuid4())[:8]
    conv = Conversation(
        id=conv_id,
        title="New conversation",
        messages=[],
    )
    st.session_state.conversations.insert(0, conv)
    st.session_state.current_conversation_id = conv_id
    st.session_state.messages = []
    persist_conversation(conv)

    return conv_id


def _save_current_conversation(*, touch_updated_at: bool = True) -> None:
    """Save current messages to the active conversation.

    ``touch_updated_at`` should be False when only switching views/history,
    so browsing old chats does not reorder the history list.
    """
    conv_id = st.session_state.get("current_conversation_id")
    if not conv_id or not st.session_state.messages:
        return
    
    for conv in st.session_state.conversations:
        if conv.id == conv_id:
            conv.messages = st.session_state.messages.copy()
            if touch_updated_at:
                conv.updated_at = datetime.now()
            persist_conversation(conv)
            break


def load_conversation(conv_id: str) -> bool:
    """Load a conversation by ID. Returns True if found."""
    # Save current first
    if st.session_state.messages and st.session_state.current_conversation_id:
        _save_current_conversation(touch_updated_at=False)
    
    for conv in st.session_state.conversations:
        if conv.id == conv_id:
            st.session_state.messages = conv.messages.copy()
            st.session_state.current_conversation_id = conv_id
            return True
    return False


def delete_conversation(conv_id: str) -> None:
    """Delete a conversation by ID (memory + disk)."""
    st.session_state.conversations = [c for c in st.session_state.conversations if c.id != conv_id]
    delete_conversation_from_disk(conv_id)

    if st.session_state.current_conversation_id == conv_id:
        st.session_state.messages = []
        st.session_state.current_conversation_id = None


def get_conversations() -> List[Conversation]:
    """Get all conversations sorted by updated_at (most recent first)."""
    return sorted(
        st.session_state.conversations,
        key=lambda c: (c.updated_at, c.created_at, c.id),
        reverse=True,
    )


def add_reasoning_step(step_num: int, title: str, content: str) -> None:
    """Add a reasoning step to the current reasoning context."""
    step = ReasoningStep(step_num=step_num, title=title, content=content)
    st.session_state.current_reasoning.append(step)


def clear_reasoning() -> None:
    """Clear the current reasoning steps."""
    st.session_state.current_reasoning = []


def update_agent_status(agent_name: str, status: str, description: str = "") -> None:
    """Update the status of a specific agent."""
    for agent in st.session_state.agents:
        if agent.name == agent_name:
            agent.status = status
            agent.last_update = datetime.now()
            if description:
                agent.description = description
            break


def reset_all_agents() -> None:
    """Reset all agents to idle status."""
    for agent in st.session_state.agents:
        agent.status = "idle"
        agent.progress = 0.0


def add_pipeline_log(message: str, level: str = "info") -> None:
    """Add a log entry to the pipeline logs."""
    st.session_state.pipeline_logs.append({
        "timestamp": datetime.now(),
        "level": level,
        "message": message,
    })


def clear_pipeline_logs() -> None:
    """Clear all pipeline logs."""
    st.session_state.pipeline_logs = []


# ── Persistent chat history ─────────────────────────────────────────────

def _ensure_history_dir() -> Path:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_history_files()
    return _HISTORY_DIR


def _migrate_legacy_history_files() -> None:
    """Auto-migrate legacy .chat_history JSON files into chat_history."""
    if not _LEGACY_HISTORY_DIR.exists() or not _LEGACY_HISTORY_DIR.is_dir():
        return

    for legacy_file in _LEGACY_HISTORY_DIR.glob("*.json"):
        target = _HISTORY_DIR / legacy_file.name
        if target.exists():
            continue
        try:
            target.write_bytes(legacy_file.read_bytes())
        except Exception:
            continue


def _conv_to_dict(conv: Conversation) -> dict:
    msgs = []
    for m in conv.messages:
        msgs.append({
            "role": m.role,
            "content": m.content,
            "timestamp": m.timestamp.isoformat(),
        })
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
        "messages": msgs,
    }


def _dict_to_conv(d: dict) -> Conversation:
    msgs = []
    for m in d.get("messages", []):
        msgs.append(ChatMessage(
            role=m["role"],
            content=m["content"],
            timestamp=datetime.fromisoformat(m.get("timestamp", datetime.now().isoformat())),
        ))
    return Conversation(
        id=d["id"],
        title=d.get("title", "Untitled"),
        messages=msgs,
        created_at=datetime.fromisoformat(d.get("created_at", datetime.now().isoformat())),
        updated_at=datetime.fromisoformat(d.get("updated_at", datetime.now().isoformat())),
    )


def persist_conversation(conv: Conversation) -> None:
    """Write a single conversation to disk as JSON."""
    d = _ensure_history_dir()
    path = d / f"{conv.id}.json"
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(_conv_to_dict(conv), f, ensure_ascii=False, indent=2)


def persist_all_conversations() -> None:
    """Flush every in-memory conversation to disk."""
    for conv in st.session_state.get("conversations", []):
        persist_conversation(conv)


def load_conversations_from_disk() -> List[Conversation]:
    """Load all conversations stored in chat_history/."""
    d = _ensure_history_dir()
    convs: List[Conversation] = []
    for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = _json.load(f)
            convs.append(_dict_to_conv(data))
        except Exception:
            continue
    return convs


def delete_conversation_from_disk(conv_id: str) -> None:
    """Remove a conversation JSON file from disk."""
    path = _HISTORY_DIR / f"{conv_id}.json"
    if path.exists():
        path.unlink()


def export_conversation_json(conv: Conversation) -> str:
    """Return a conversation as a formatted JSON string."""
    return _json.dumps(_conv_to_dict(conv), ensure_ascii=False, indent=2)


def list_history_json_files() -> List[Path]:
    """Return detected chat history JSON files from chat_history/."""
    d = _ensure_history_dir()
    return sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)


def import_conversation_json(payload: str) -> Conversation:
    """Import one conversation from JSON text and persist it."""
    data = _json.loads(payload)
    conv = _dict_to_conv(_normalize_import_payload(data))

    existing_ids = {c.id for c in st.session_state.get("conversations", [])}
    if conv.id in existing_ids:
        conv.id = str(uuid.uuid4())[:8]
        conv.updated_at = datetime.now()

    st.session_state.conversations.insert(0, conv)
    st.session_state.current_conversation_id = conv.id
    st.session_state.messages = conv.messages.copy()
    persist_conversation(conv)
    return conv


def _normalize_import_payload(data: dict) -> dict:
    """Validate and normalize imported JSON to conversation schema."""
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object.")

    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Conversation JSON must contain a non-empty 'messages' array.")

    normalized_messages = []
    for i, msg in enumerate(messages, start=1):
        if not isinstance(msg, dict):
            raise ValueError(f"Message #{i} must be an object.")
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content")
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Message #{i} has invalid role '{role}'.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Message #{i} must include non-empty string 'content'.")
        normalized_messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": msg.get("timestamp", datetime.now().isoformat()),
            }
        )

    return {
        "id": str(data.get("id") or str(uuid.uuid4())[:8]),
        "title": str(data.get("title") or "Imported conversation"),
        "created_at": data.get("created_at", datetime.now().isoformat()),
        "updated_at": data.get("updated_at", datetime.now().isoformat()),
        "messages": normalized_messages,
    }
