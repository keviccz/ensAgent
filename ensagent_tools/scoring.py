"""
Tool: run the Scoring pipeline (Stage B).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Dict

from ensagent_tools.config_manager import PipelineConfig


def run_scoring(
    cfg: PipelineConfig,
    *,
    input_dir: str = "",
    output_dir: str = "",
    vlm_off: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> Dict[str, Any]:
    """Execute ``scoring/scoring.py``."""
    repo = cfg.repo_root()
    script = repo / "scoring" / "scoring.py"
    if not script.exists():
        return {"ok": False, "error": f"Not found: {script}"}

    cmd = [sys.executable, str(script)]
    if input_dir:
        cmd += ["--input_dir", str(input_dir)]
    if output_dir:
        cmd += ["--output_dir", str(output_dir)]
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
    effective_temperature = cfg.temperature if temperature is None else float(temperature)
    effective_top_p = cfg.top_p if top_p is None else float(top_p)
    cmd += ["--temperature", str(effective_temperature)]
    cmd += ["--top_p", str(effective_top_p)]
    effective_vlm_off = cfg.vlm_off if vlm_off is None else bool(vlm_off)
    if effective_vlm_off:
        cmd.append("--vlm_off")

    print(f"[Tool] Running scoring")
    p = subprocess.run(cmd, cwd=str(script.parent), check=False)
    return {
        "ok": p.returncode == 0,
        "exit_code": p.returncode,
        "temperature_used": float(effective_temperature),
        "top_p_used": float(effective_top_p),
        "vlm_off_used": bool(effective_vlm_off),
    }
