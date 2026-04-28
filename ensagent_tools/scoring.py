"""
Tool: run the Scoring pipeline (Stage B).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from ensagent_tools.config_manager import PipelineConfig
from ensagent_tools.subprocess_stream import (
    CancelCheck,
    ProgressCallback,
    run_subprocess_streaming,
)


def _normalize_pic_sample_slug(sample_id: str | None) -> str:
    value = str(sample_id or "").strip()
    if not value:
        return ""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")


def _pic_scores_file(pic_dir: Path, sample_id: str | None) -> Path:
    slug = _normalize_pic_sample_slug(sample_id)
    filename = "all_domains_scores.json" if not slug else f"all_domains_scores_{slug}.json"
    return pic_dir / "output" / filename


def _emit_progress(progress_callback: ProgressCallback | None, payload: Dict[str, Any]) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(payload)
    except Exception:
        return


def _run_command(
    *,
    cmd: list[str],
    cwd: Path,
    tool: str,
    stage: str,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
    env: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    if progress_callback is None and cancel_check is None:
        p = subprocess.run(cmd, cwd=str(cwd), check=False, env=env)
        return {
            "returncode": int(p.returncode),
            "interrupted": False,
            "log_line_count": 0,
            "stdout_tail": [],
        }
    return run_subprocess_streaming(
        cmd=cmd,
        cwd=cwd,
        tool=tool,
        stage=stage,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        env=env,
    )


def run_scoring(
    cfg: PipelineConfig,
    *,
    input_dir: str = "",
    output_dir: str = "",
    vlm_off: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_degs: int | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> Dict[str, Any]:
    """Execute ``scoring/scoring.py``."""
    repo = cfg.repo_root()
    script = repo / "scoring" / "scoring.py"
    if not script.exists():
        return {"ok": False, "error": f"Not found: {script}"}

    effective_input_dir = input_dir or str(cfg.resolved_scoring_input_dir())
    effective_output_dir = output_dir or str(cfg.resolved_scoring_output_dir())
    subprocess_env = dict(os.environ)
    subprocess_env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess_env.setdefault("PYTHONUTF8", "1")
    cmd = [sys.executable, str(script)]
    cmd += ["--input_dir", str(effective_input_dir)]
    cmd += ["--output_dir", str(effective_output_dir)]
    if cfg.sample_id:
        cmd += ["--sample_id", str(cfg.sample_id)]
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
    effective_top_degs = cfg.top_degs if top_degs is None else int(top_degs)
    cmd += ["--top_n_deg", str(effective_top_degs)]

    pic_dir = script.parent / "pic_analyze"
    pic_scores_file = _pic_scores_file(pic_dir, cfg.sample_id)
    pic_analyze_autorun = False
    pic_analyze_result: Dict[str, Any] | None = None
    if (not effective_vlm_off) and (not pic_scores_file.exists()):
        pic_script = pic_dir / "run.py"
        if not pic_script.exists():
            return {
                "ok": False,
                "error": (
                    f"Visual score file missing: {pic_scores_file}. "
                    f"Auto-run script not found: {pic_script}"
                ),
            }

        pic_analyze_autorun = True
        _emit_progress(
            progress_callback,
            {
                "kind": "tool_log",
                "tool": "run_scoring",
                "stage": "scoring",
                "level": "warning",
                "message": (
                    "Visual score file not found; running pic_analyze first: "
                    f"{pic_scores_file}"
                ),
            },
        )
        pic_cmd = [sys.executable, str(pic_script)]
        if cfg.sample_id:
            pic_cmd += ["--sample_id", str(cfg.sample_id)]
        pic_analyze_result = _run_command(
            cmd=pic_cmd,
            cwd=pic_script.parent,
            tool="run_scoring",
            stage="scoring_visual",
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            env=subprocess_env,
        )
        pic_exit = int(pic_analyze_result.get("returncode", 1))
        pic_interrupted = bool(pic_analyze_result.get("interrupted", False))
        if pic_interrupted:
            return {
                "ok": False,
                "exit_code": pic_exit,
                "interrupted": True,
                "error": "Scoring interrupted while running pic_analyze pre-step.",
                "pic_analyze_autorun": True,
                "pic_analyze_exit_code": pic_exit,
                "log_tail": pic_analyze_result.get("stdout_tail", []),
            }
        if pic_exit != 0 or not pic_scores_file.exists():
            return {
                "ok": False,
                "exit_code": pic_exit,
                "error": (
                    "Visual score file missing and pic_analyze pre-step failed. "
                    f"Expected file: {pic_scores_file}"
                ),
                "pic_analyze_autorun": True,
                "pic_analyze_exit_code": pic_exit,
                "log_tail": pic_analyze_result.get("stdout_tail", []),
            }

    print(f"[Tool] Running scoring")
    run_result = _run_command(
        cmd=cmd,
        cwd=script.parent,
        tool="run_scoring",
        stage="scoring",
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        env=subprocess_env,
    )
    exit_code = int(run_result.get("returncode", 1))
    interrupted = bool(run_result.get("interrupted", False))
    return {
        "ok": (exit_code == 0 and not interrupted),
        "exit_code": exit_code,
        "interrupted": interrupted,
        "input_dir_used": str(effective_input_dir),
        "output_dir_used": str(effective_output_dir),
        "temperature_used": float(effective_temperature),
        "top_p_used": float(effective_top_p),
        "vlm_off_used": bool(effective_vlm_off),
        "top_degs_used": int(effective_top_degs),
        "pic_analyze_autorun": pic_analyze_autorun,
        "pic_analyze_exit_code": (
            int(pic_analyze_result.get("returncode", 0))
            if isinstance(pic_analyze_result, dict)
            else None
        ),
        "visual_scores_file": str(pic_scores_file),
        "log_tail": run_result.get("stdout_tail", []),
        "log_line_count": int(run_result.get("log_line_count", 0)),
    }
