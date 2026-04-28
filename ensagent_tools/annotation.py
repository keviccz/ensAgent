"""
Tool: run multi-agent annotation (Stage D).
"""

from __future__ import annotations

import os
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


def _run_command(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict | None = None,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> Dict[str, Any]:
    merged_env = {**os.environ, **(env or {})}
    if progress_callback is None and cancel_check is None:
        p = subprocess.run(cmd, cwd=str(cwd), env=merged_env, check=False)
        return {
            "returncode": int(p.returncode),
            "interrupted": False,
            "log_line_count": 0,
            "stdout_tail": [],
        }
    return run_subprocess_streaming(
        cmd=cmd,
        cwd=cwd,
        tool="run_annotation",
        stage="annotation",
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        env=merged_env,
    )


def _candidate_best_files(sample_id: str) -> dict[str, tuple[str, ...]]:
    return {
        "spot": (
            f"BEST_{sample_id}_spot.csv",
            f"BEST_DLPFC_{sample_id}_spot.csv",
        ),
        "DEGs": (
            f"BEST_{sample_id}_DEGs.csv",
            f"BEST_DLPFC_{sample_id}_DEGs.csv",
        ),
        "PATHWAY": (
            f"BEST_{sample_id}_PATHWAY.csv",
            f"BEST_DLPFC_{sample_id}_PATHWAY.csv",
        ),
    }


def run_annotation(
    cfg: PipelineConfig,
    *,
    data_dir: str = "",
    sample_id: str = "",
    domain: str = "",
    output_dir: str = "",
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> Dict[str, Any]:
    """Execute annotation via ``annotation/run_annotation_main.py`` (standalone entry-point)."""
    repo = cfg.repo_root()
    script = repo / "annotation" / "run_annotation_main.py"
    if not script.exists():
        return {"ok": False, "error": f"Not found: {script}"}

    sid = sample_id or cfg.sample_id
    dd = data_dir or str(cfg.resolved_best_output_dir())

    if not sid or not dd:
        return {"ok": False, "error": "data_dir and sample_id are required"}

    data_dir_path = Path(dd)
    if not data_dir_path.exists():
        return {"ok": False, "error": f"annotation data_dir not found: {data_dir_path}"}
    out_dir = output_dir or str(data_dir_path / "annotation_output")

    missing_labels: list[str] = []
    for label, candidates in _candidate_best_files(str(sid)).items():
        if not any((data_dir_path / name).exists() for name in candidates):
            missing_labels.append(label)
    if missing_labels:
        return {
            "ok": False,
            "error": (
                "Missing BEST artifacts required for annotation in "
                f"{data_dir_path}: {', '.join(missing_labels)}"
            ),
        }

    cmd = [
        sys.executable, str(script),
        "--data_dir", str(dd),
        "--sample_id", str(sid),
        "--output_dir", str(out_dir),
    ]
    if cfg.api_provider:
        cmd += ["--api_provider", str(cfg.api_provider)]
    if cfg.api_key:
        cmd += ["--api_key", str(cfg.api_key)]
    if cfg.api_endpoint:
        cmd += ["--api_endpoint", str(cfg.api_endpoint)]
    if cfg.api_version:
        cmd += ["--api_version", str(cfg.api_version)]
    if cfg.api_model or cfg.api_deployment:
        cmd += ["--api_model", str(cfg.api_model or cfg.api_deployment)]
    if domain:
        cmd += ["--domain", str(domain)]

    # Ensure repo root is on PYTHONPATH so `annotation` package is importable
    existing_pypath = os.environ.get("PYTHONPATH", "")
    repo_str = str(repo)
    new_pypath = f"{repo_str}{os.pathsep}{existing_pypath}" if existing_pypath else repo_str
    extra_env = {"PYTHONPATH": new_pypath}

    print(f"[Tool] Running multi-agent annotation -> {dd}")
    run_result = _run_command(
        cmd=cmd,
        cwd=repo,
        env=extra_env,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    exit_code = int(run_result.get("returncode", 1))
    interrupted = bool(run_result.get("interrupted", False))
    return {
        "ok": (exit_code == 0 and not interrupted),
        "exit_code": exit_code,
        "interrupted": interrupted,
        "data_dir": dd,
        "output_dir": out_dir,
        "log_tail": run_result.get("stdout_tail", []),
        "log_line_count": int(run_result.get("log_line_count", 0)),
    }
