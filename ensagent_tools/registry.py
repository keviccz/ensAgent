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
from ensagent_tools.ari_analysis import compute_ari
from ensagent_tools.python_use import python_use

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
                    "n_clusters": {"type": "integer", "description": "Number of clusters."},
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
            "description": "Run Scoring (Stage B): LLM evaluation and consensus matrix generation. Defaults to config csv_path or scoring/input.",
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
            "name": "python_use",
            "description": (
                "Execute a Python script file or inline Python code in the ensagent environment. "
                "Use this to run any .py script in the project (e.g. scoring/pic_analyze/auto_analyzer.py) "
                "or to execute ad-hoc data processing code. "
                "Returns stdout, stderr, and exit code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Path to a .py file (relative to repo root, or absolute).",
                    },
                    "module": {
                        "type": "string",
                        "description": "Python module path for scripts with relative imports, e.g. 'scoring.pic_analyze.auto_analyzer'. Runs as: python -m <module>.",
                    },
                    "code": {
                        "type": "string",
                        "description": "Inline Python source code to execute.",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "CLI arguments passed to the script (only used with 'script').",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (default: repo root).",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max execution time in seconds (default: 120).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_ari",
            "description": (
                "Compute ARI (Adjusted Rand Index) against ground truth and generate a spatial "
                "clustering plot. All paths are auto-detected from the current pipeline config. "
                "Use after Stage B (run_scoring) has produced scores_matrix.csv."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_id":       {"type": "string", "description": "Sample ID (default: from config)."},
                    "scores_dir":      {"type": "string", "description": "Directory with scores_matrix.csv + labels_matrix.csv (default: scoring/output/<sample_id>/consensus)."},
                    "visium_dir":      {"type": "string", "description": "Visium data directory (default: data_path in config)."},
                    "output_dir":      {"type": "string", "description": "Output directory for the PNG (default: output/ari/<sample_id>)."},
                    "truth_file":      {"type": "string", "description": "Path to metadata.tsv with ground-truth labels (auto-searched if omitted)."},
                    "apply_smoothing": {"type": "boolean", "description": "Apply multi-scale spatial smoothing (default: true)."},
                },
                "required": [],
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
    {
        "type": "function",
        "function": {
            "name": "get_cluster_image",
            "description": (
                "Return the spatial clustering visualization image for a sample as a base64 PNG. "
                "Use when the user asks to see, show, or visualize the clustering result or cluster map. "
                "The image will be displayed directly in the chat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_id": {
                        "type": "string",
                        "description": "Sample identifier, e.g. '151507' or 'DLPFC_151507'. Defaults to current config.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_annotation_result",
            "description": (
                "Read existing annotation results from output files WITHOUT running the pipeline. "
                "Use this when the user asks about domain annotations, labels, marker genes, or "
                "cell types for a specific sample. "
                "Do NOT use run_annotation — that triggers the full Stage D pipeline (takes hours)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_id": {
                        "type": "string",
                        "description": "Sample identifier, e.g. '151507' or 'DLPFC_151507'.",
                    },
                    "domain_id": {
                        "type": "integer",
                        "description": "Domain/cluster ID to retrieve. Use -1 to get all domains.",
                        "default": -1,
                    },
                },
                "required": ["sample_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_cluster_scatter",
            "description": (
                "Show an interactive spatial cluster scatter plot in the chat. "
                "Use when the user asks to visualize clusters interactively, click domains, "
                "or explore spatial distribution. Unlike get_cluster_image (static PNG), "
                "this renders a clickable canvas where each spot can be selected to show its annotation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_id": {
                        "type": "string",
                        "description": "Sample identifier, e.g. 'DLPFC_151507'. Defaults to current config.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_scores_matrix",
            "description": (
                "Show the Scores Matrix visualization in the chat. "
                "Displays per-method, per-domain LLM evaluation scores as a heatmap table. "
                "Use when user asks about scoring results, method performance, or score breakdown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_id": {
                        "type": "string",
                        "description": "Sample identifier. Defaults to current config.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_distributions",
            "description": (
                "Show Domain Distribution and Method Ranking charts in the chat (interactive). "
                "Domain Distribution shows spot counts per domain; Method Ranking shows "
                "average LLM scores per clustering method, filterable by domain. "
                "Use when user asks about domain sizes, spot counts, or method comparison."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_id": {
                        "type": "string",
                        "description": "Sample identifier. Defaults to current config.",
                    },
                },
                "required": [],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Dispatch table: name -> callable(cfg, **kwargs) -> dict
# ---------------------------------------------------------------------------

def _get_cluster_image(cfg: PipelineConfig, **kw: Any) -> Dict[str, Any]:
    """Return the spatial clustering PNG as base64 for display in chat."""
    import base64
    from pathlib import Path

    sample_id = str(kw.get("sample_id") or cfg.sample_id or "").strip()
    repo_root = cfg.repo_root()

    variants = [sample_id]
    if "_" in sample_id:
        variants.append(sample_id.split("_", 1)[1])

    candidates: List[Path] = []
    for v in variants:
        candidates.extend([
            repo_root / "output" / "ari" / v / f"{v}_clustering.png",
            repo_root / "output" / "best" / v / f"{v}_result.png",
            repo_root / "Tool-runner" / v / f"{v}_result.png",
        ])
    # Glob scan example_data and output for any clustering PNG
    example_png = repo_root / "example_data"
    if example_png.exists():
        for p in sorted(example_png.glob("*.png")):
            candidates.append(p)
    for base in [repo_root / "output", repo_root / "Tool-runner"]:
        if base.exists():
            for p in base.rglob("*clustering*.png"):
                candidates.append(p)
            for p in base.rglob("*result*.png"):
                candidates.append(p)

    for p in candidates:
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return {
                "ok": True,
                "sample_id": sample_id,
                "image_b64": b64,
                "source": str(p),
                "message": f"Cluster visualization for {sample_id or 'current sample'}.",
            }

    return {
        "ok": False,
        "error": (
            f"No cluster image found for sample '{sample_id}'. "
            "Run Stage A (Tool-Runner) or Stage B (Scoring with ARI) to generate one."
        ),
    }


def _show_cluster_scatter(cfg: PipelineConfig, **kw: Any) -> Dict[str, Any]:
    """Return chart token for interactive spatial scatter in chat."""
    sample_id = str(kw.get("sample_id") or cfg.sample_id or "").strip()
    return {
        "ok": True,
        "sample_id": sample_id,
        "_chart": {"type": "scatter", "sampleId": sample_id or "DLPFC_151507"},
        "message": f"Interactive cluster scatter for {sample_id or 'current sample'} — rendered in chat.",
    }


def _show_scores_matrix(cfg: PipelineConfig, **kw: Any) -> Dict[str, Any]:
    """Return chart token for Scores Matrix in chat."""
    sample_id = str(kw.get("sample_id") or cfg.sample_id or "").strip()
    return {
        "ok": True,
        "sample_id": sample_id,
        "_chart": {"type": "scores_matrix", "sampleId": sample_id or "DLPFC_151507"},
        "message": f"Scores Matrix for {sample_id or 'current sample'} — rendered in chat.",
    }


def _show_distributions(cfg: PipelineConfig, **kw: Any) -> Dict[str, Any]:
    """Return chart token for Domain Distribution + Method Ranking in chat."""
    sample_id = str(kw.get("sample_id") or cfg.sample_id or "").strip()
    return {
        "ok": True,
        "sample_id": sample_id,
        "_chart": {"type": "distributions", "sampleId": sample_id or "DLPFC_151507"},
        "message": f"Domain Distribution and Method Ranking for {sample_id or 'current sample'} — rendered in chat.",
    }


def _get_annotation_result(cfg: PipelineConfig, **kw: Any) -> Dict[str, Any]:
    """Read existing annotation results from output files without running the pipeline."""
    import json
    from pathlib import Path

    sample_id = str(kw.get("sample_id") or cfg.sample_id or "").strip()
    domain_id = kw.get("domain_id", -1)

    repo_root = cfg.repo_root()
    variants = [sample_id]
    if "_" in sample_id:
        variants.append(sample_id.split("_", 1)[1])

    candidates: List[Path] = []
    for v in variants:
        candidates.extend([
            repo_root / "output" / "best" / v / "annotation_output" / "domain_annotations.json",
            repo_root / "output" / "annotation_runs" / v / "domain_annotations.json",
            repo_root / "output" / "best" / v / f"{v}_annotation.json",
        ])
    candidates.append(repo_root / "output" / "best" / "annotation_output" / "domain_annotations.json")

    data = None
    found_path = None
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            found_path = str(p)
            break

    if data is None:
        return {
            "ok": False,
            "error": (
                f"No annotation results found for sample '{sample_id}'. "
                "Run Stage D (Annotation) first, or check that the sample_id is correct."
            ),
        }

    if isinstance(data, list):
        if int(domain_id) == -1:
            return {"ok": True, "sample_id": sample_id, "source": found_path, "annotations": data}
        for item in data:
            if item.get("domain_id") == int(domain_id):
                return {"ok": True, "sample_id": sample_id, "domain_id": int(domain_id),
                        "source": found_path, "annotation": item}
        return {"ok": False, "error": f"Domain {domain_id} not found in annotation results."}

    # dict format {"1": {...}}
    if int(domain_id) == -1:
        return {"ok": True, "sample_id": sample_id, "source": found_path, "annotations": data}
    key = str(domain_id)
    if key not in data:
        return {"ok": False, "error": f"Domain {domain_id} not found in annotation results."}
    return {"ok": True, "sample_id": sample_id, "domain_id": int(domain_id),
            "source": found_path, "annotation": data[key]}


TOOL_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "check_envs": lambda cfg, **kw: check_envs(cfg),
    "setup_envs": lambda cfg, **kw: setup_envs(cfg),
    "run_tool_runner": lambda cfg, **kw: run_tool_runner(cfg, **kw),
    "run_scoring": lambda cfg, **kw: run_scoring(cfg, **kw),
    "run_best_builder": lambda cfg, **kw: run_best_builder(cfg, **kw),
    "run_annotation": lambda cfg, **kw: run_annotation(cfg, **kw),
    "python_use":     lambda cfg, **kw: python_use(cfg, **kw),
    "compute_ari":    lambda cfg, **kw: compute_ari(cfg, **kw),
    "run_end_to_end": lambda cfg, **kw: run_full_pipeline(cfg, **kw),
    "show_config": lambda cfg, **kw: show_config(cfg),
    "set_config": lambda cfg, **kw: set_config_value(
        key=kw.get("name", ""), value=kw.get("value", ""),
    ),
    "get_annotation_result": _get_annotation_result,
    "get_cluster_image": _get_cluster_image,
    "show_cluster_scatter": _show_cluster_scatter,
    "show_scores_matrix": _show_scores_matrix,
    "show_distributions": _show_distributions,
}


def _is_progress_kwarg_mismatch(exc: TypeError) -> bool:
    msg = str(exc)
    if "unexpected keyword argument" not in msg:
        return False
    return ("progress_callback" in msg) or ("cancel_check" in msg)


def execute_tool(
    name: str,
    args: Dict[str, Any],
    cfg: PipelineConfig,
    *,
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Dict[str, Any]:
    """Look up *name* in the registry and call it with *cfg* + *args*."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        effective_args = dict(args or {})
        if progress_callback is not None:
            effective_args["progress_callback"] = progress_callback
        if cancel_check is not None:
            effective_args["cancel_check"] = cancel_check
        return fn(cfg, **effective_args)
    except TypeError as e:
        if not _is_progress_kwarg_mismatch(e):
            return {"ok": False, "error": str(e)}
        compat_args = dict(effective_args)
        compat_args.pop("progress_callback", None)
        compat_args.pop("cancel_check", None)
        try:
            out = fn(cfg, **compat_args)
        except Exception as retry_exc:
            return {"ok": False, "error": str(retry_exc)}
        if isinstance(out, dict):
            patched = dict(out)
            patched.setdefault("compat_retry_used", True)
            patched.setdefault("compat_retry_reason", "legacy_tool_signature_missing_progress_callback_or_cancel_check")
            return patched
        return {
            "ok": True,
            "result": out,
            "compat_retry_used": True,
            "compat_retry_reason": "legacy_tool_signature_missing_progress_callback_or_cancel_check",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
