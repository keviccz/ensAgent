"""
ensagent_tools -- unified tool layer for the EnsAgent pipeline.

Every pipeline capability (tool-runner, scoring, BEST builder, annotation,
environment management, configuration) is exposed here as a plain Python
function **and** as an OpenAI-compatible function-calling JSON schema.

Consumers:
  - ``ensagent_agent/chat.py``  (CLI LLM agent)
  - ``streamlit_app/``          (web UI)
  - ``endtoend.py``             (CLI pipeline runner)
"""

from ensagent_tools.config_manager import PipelineConfig, load_config, save_config
from ensagent_tools.registry import TOOL_REGISTRY, TOOL_SCHEMAS, execute_tool

__all__ = [
    "PipelineConfig",
    "load_config",
    "save_config",
    "TOOL_REGISTRY",
    "TOOL_SCHEMAS",
    "execute_tool",
]
