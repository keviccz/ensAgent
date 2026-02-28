"""
Tool: run multi-agent annotation (Stage D).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Dict

from ensagent_tools.config_manager import PipelineConfig


def run_annotation(
    cfg: PipelineConfig,
    *,
    data_dir: str = "",
    sample_id: str = "",
    domain: str = "",
) -> Dict[str, Any]:
    """Execute annotation via ``scoring/scoring.py --annotation_multiagent``."""
    repo = cfg.repo_root()
    script = repo / "scoring" / "scoring.py"
    if not script.exists():
        return {"ok": False, "error": f"Not found: {script}"}

    sid = sample_id or cfg.sample_id
    dd = data_dir or str(cfg.resolved_best_output_dir())

    if not sid or not dd:
        return {"ok": False, "error": "data_dir and sample_id are required"}

    cmd = [
        sys.executable, str(script),
        "--annotation_multiagent",
        "--annotation_data_dir", str(dd),
        "--annotation_sample_id", str(sid),
    ]
    if cfg.api_provider:
        cmd += ["--api_provider", str(cfg.api_provider)]
    if cfg.api_key:
        cmd += ["--api_key", str(cfg.api_key)]
    if cfg.api_endpoint:
        cmd += ["--api_endpoint", str(cfg.api_endpoint)]
    if cfg.api_version:
        cmd += ["--api_version", str(cfg.api_version)]
    if (cfg.api_model or cfg.api_deployment):
        cmd += ["--api_model", str(cfg.api_model or cfg.api_deployment)]
    if (cfg.api_provider or "").strip().lower() == "azure":
        if cfg.api_key:
            cmd += ["--openai_key", str(cfg.api_key)]
        if cfg.api_endpoint:
            cmd += ["--azure_endpoint", str(cfg.api_endpoint)]
        if cfg.api_version:
            cmd += ["--azure_api_version", str(cfg.api_version)]
        if (cfg.api_model or cfg.api_deployment):
            cmd += ["--azure_deployment", str(cfg.api_model or cfg.api_deployment)]
    if domain:
        cmd += ["--domain", str(domain)]

    print(f"[Tool] Running multi-agent annotation -> {dd}")
    p = subprocess.run(cmd, cwd=str(script.parent), check=False)
    return {"ok": p.returncode == 0, "exit_code": p.returncode, "data_dir": dd}
