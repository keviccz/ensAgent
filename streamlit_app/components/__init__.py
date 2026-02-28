"""EnsAgent UI Components.

Lazy exports are used to avoid importing all Streamlit-bound modules eagerly.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "render_sidebar": ("sidebar", "render_sidebar"),
    "render_overview": ("dashboard", "render_overview"),
    "render_spatial_analysis": ("dashboard", "render_spatial_analysis"),
    "render_agent_status": ("agents", "render_agent_status"),
    "render_agent_orchestrator": ("agents", "render_agent_orchestrator"),
    "render_thinking_log": ("thinking_log", "render_thinking_log"),
    "render_reasoning_panel": ("thinking_log", "render_reasoning_panel"),
    "render_chat_interface": ("chat", "render_chat_interface"),
    "render_settings": ("settings", "render_settings"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
