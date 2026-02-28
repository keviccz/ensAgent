"""
Pipeline configuration: load / save / merge with CLI overrides.

The canonical config file is ``<repo_root>/pipeline_config.yaml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "pipeline_config.yaml"
_EXAMPLE_CONFIG_PATH = _REPO_ROOT / "pipeline_config.example.yaml"


@dataclass
class PipelineConfig:
    """All parameters needed by the end-to-end pipeline."""

    # Required
    data_path: str = ""
    sample_id: str = ""

    # Phase toggles
    skip_tool_runner: bool = False
    skip_scoring: bool = False
    run_best: bool = True
    run_annotation_multiagent: bool = True

    # Tool-runner
    n_clusters: int = 7
    random_seed: int = 2023
    methods: List[str] = field(default_factory=lambda: [
        "IRIS", "BASS", "DR-SC", "BayesSpace",
        "SEDR", "GraphST", "STAGATE", "stLearn",
    ])

    # Path overrides (empty → auto-derived)
    tool_output_dir: str = ""
    best_output_dir: str = ""

    # Scoring
    overwrite_staging: bool = False
    vlm_off: bool = False
    temperature: float = 0.7
    top_p: float = 1.0

    # BEST builder
    best_smooth_knn: bool = False
    best_truth_file: str = ""

    # Environment
    conda_exe: str = "mamba"
    env_names: Dict[str, str] = field(default_factory=lambda: {
        "R": "R", "PY": "PY", "PY2": "PY2",
    })

    # Generic API settings (used by Streamlit Settings)
    api_provider: str = ""
    api_key: str = ""
    api_endpoint: str = ""
    api_version: str = ""
    api_model: str = ""
    api_deployment: str = ""

    # Azure OpenAI (optional; prefer env vars)
    azure_openai_key: str = ""
    azure_endpoint: str = ""
    azure_deployment: str = "gpt-4o"
    azure_api_version: str = "2024-12-01-preview"

    # ---- helpers ----

    def repo_root(self) -> Path:
        return _REPO_ROOT

    def resolved_tool_output_dir(self) -> Path:
        if self.tool_output_dir:
            return Path(self.tool_output_dir)
        return _REPO_ROOT / "output" / "tool_runner" / self.sample_id

    def resolved_best_output_dir(self) -> Path:
        if self.best_output_dir:
            return Path(self.best_output_dir)
        return _REPO_ROOT / "output" / "best" / self.sample_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(
    path: str | Path | None = None,
    *,
    cli_overrides: Dict[str, Any] | None = None,
) -> PipelineConfig:
    """Load config from YAML, then overlay *cli_overrides* (non-None values win)."""
    path = Path(path) if path else _DEFAULT_CONFIG_PATH
    raw: Dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    # Flatten env_names if present
    cfg = PipelineConfig(**{
        k: v for k, v in raw.items()
        if k in PipelineConfig.__dataclass_fields__ and v is not None
    })

    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)

    return cfg


def save_config(cfg: PipelineConfig, path: str | Path | None = None) -> Path:
    """Persist *cfg* to YAML."""
    path = Path(path) if path else _DEFAULT_CONFIG_PATH
    data = cfg.to_dict()
    # Strip empty secrets so they don't get written to disk
    for secret_key in ("api_key", "azure_openai_key"):
        if not data.get(secret_key):
            data.pop(secret_key, None)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def show_config(cfg: PipelineConfig) -> Dict[str, Any]:
    """Return a display-safe summary (masks secrets)."""
    d = cfg.to_dict()
    if d.get("api_key"):
        d["api_key"] = d["api_key"][:4] + "****"
    if d.get("azure_openai_key"):
        d["azure_openai_key"] = d["azure_openai_key"][:4] + "****"
    return {"ok": True, "config": d}


def set_config_value(key: str, value: Any, path: str | Path | None = None) -> Dict[str, Any]:
    """Update a single key in the persisted YAML config."""
    cfg = load_config(path)
    if not hasattr(cfg, key):
        return {"ok": False, "error": f"Unknown config key: {key}"}
    # coerce types
    field_type = type(getattr(cfg, key))
    try:
        if field_type is bool:
            coerced = str(value).lower() in ("true", "1", "yes")
        elif field_type is int:
            coerced = int(value)
        elif field_type is list:
            coerced = value if isinstance(value, list) else [s.strip() for s in str(value).split(",")]
        else:
            coerced = str(value)
    except Exception as e:
        return {"ok": False, "error": f"Cannot coerce {value!r} to {field_type.__name__}: {e}"}
    setattr(cfg, key, coerced)
    save_config(cfg, path)
    return {"ok": True, "updated": {key: coerced}}
