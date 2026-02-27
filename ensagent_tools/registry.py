"""
Unified tool registry: JSON schemas (OpenAI function-calling format) + dispatch.

Usage::

    from ensagent_tools import TOOL_SCHEMAS, execute_tool, load_config

    cfg = load_config()
    result = execute_tool("run_tool_runner", {"data_path": "...", "sample_id": "..."}, cfg)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from ensagent_tools.config_manager import (
    PipelineConfig,
    show_config,
    set_config_value,
)
from ensagent_tools.env_manager import check_envs, setup_envs
from ensagent_tools.tool_runner import run_tool_runner
from ensagent_tools.scoring import run_scoring
from ensagent_tools.best_builder import run_best_builder
from ensagent_tools.annotation import run_annotation
from ensagent_tools.pipeline import run_full_pipeline

# ---------------------------------------------------------------------------
# Schema definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_envs",
            "description": "Check whether required conda/mamba environments (R, PY, PY2) exist.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "setup_envs",
            "description": "Create missing conda/mamba environments from envs/*.yml files.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tool_runner",
            "description": "Run Tool-runner (Stage A): execute spatial clustering methods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to Visium data directory."},
                    "sample_id": {"type": "string", "description": "Sample identifier."},
                    "n_clusters": {"type": "integer", "description": "Number of clusters.", "default": 7},
                    "methods": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Clustering methods to run (default: all 8).",
                    },
                },
                "required": ["data_path", "sample_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_scoring",
            "description": "Run Scoring (Stage B): LLM evaluation and consensus matrix generation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_dir": {"type": "string", "description": "Scoring input directory."},
                    "output_dir": {"type": "string", "description": "Scoring output directory."},
                    "vlm_off": {
                        "type": "boolean",
                        "description": "Disable visual-score integration from pic_analyze output.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_best_builder",
            "description": "Run BEST builder (Stage C): select optimal domain labels and generate BEST_* files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string", "description": "Sample identifier."},
                    "scores_matrix": {"type": "string", "description": "Path to scores_matrix.csv."},
                    "labels_matrix": {"type": "string", "description": "Path to labels_matrix.csv."},
                    "spot_template": {"type": "string", "description": "Path to a *_spot.csv template file."},
                    "visium_dir": {"type": "string", "description": "Visium data directory for DEG computation."},
                    "output_dir": {"type": "string", "description": "Output directory for BEST_* files."},
                    "smooth_knn": {"type": "boolean", "description": "Enable kNN spatial smoothing."},
                    "truth_file": {"type": "string", "description": "Optional truth file for ARI."},
                },
                "required": ["sample_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_annotation",
            "description": "Run multi-agent annotation (Stage D): annotate spatial domains.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string", "description": "Directory containing BEST_* files."},
                    "sample_id": {"type": "string", "description": "Sample identifier."},
                    "domain": {"type": "string", "description": "Specific domain IDs to annotate (comma-separated)."},
                },
                "required": ["sample_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_end_to_end",
            "description": "Run the complete end-to-end pipeline (Tool-runner -> Scoring -> BEST -> Annotation).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_config",
            "description": "Show the current pipeline configuration from pipeline_config.yaml.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_config",
            "description": "Update a configuration parameter in pipeline_config.yaml.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Configuration key to update."},
                    "value": {"type": "string", "description": "New value."},
                },
                "required": ["name", "value"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Dispatch table: name -> callable(cfg, **kwargs) -> dict
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "check_envs": lambda cfg, **kw: check_envs(cfg),
    "setup_envs": lambda cfg, **kw: setup_envs(cfg),
    "run_tool_runner": lambda cfg, **kw: run_tool_runner(cfg, **kw),
    "run_scoring": lambda cfg, **kw: run_scoring(cfg, **kw),
    "run_best_builder": lambda cfg, **kw: run_best_builder(cfg, **kw),
    "run_annotation": lambda cfg, **kw: run_annotation(cfg, **kw),
    "run_end_to_end": lambda cfg, **kw: run_full_pipeline(cfg),
    "show_config": lambda cfg, **kw: show_config(cfg),
    "set_config": lambda cfg, **kw: set_config_value(
        key=kw.get("name", ""), value=kw.get("value", ""),
    ),
}


def execute_tool(name: str, args: Dict[str, Any], cfg: PipelineConfig) -> Dict[str, Any]:
    """Look up *name* in the registry and call it with *cfg* + *args*."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return fn(cfg, **args)
    except Exception as e:
        return {"ok": False, "error": str(e)}
