"""
Tool: run the BEST builder (Stage C).
"""

from __future__ import annotations

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
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> Dict[str, Any]:
    if progress_callback is None and cancel_check is None:
        p = subprocess.run(cmd, cwd=str(cwd), check=False)
        return {
            "returncode": int(p.returncode),
            "interrupted": False,
            "log_line_count": 0,
            "stdout_tail": [],
        }
    return run_subprocess_streaming(
        cmd=cmd,
        cwd=cwd,
        tool="run_best_builder",
        stage="best_builder",
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )


def _resolve_spot_template(cfg: PipelineConfig, *, sample_id: str, repo: Path) -> tuple[str | None, list[Path]]:
    """Resolve BEST spot template from configured scoring input locations."""
    search_dirs: list[Path] = []
    cfg_input = cfg.resolved_scoring_input_dir()
    search_dirs.append(cfg_input)
    default_input = repo / "scoring" / "input"
    if default_input != cfg_input:
        search_dirs.append(default_input)

    for input_dir in search_dirs:
        if not input_dir.exists() or not input_dir.is_dir():
            continue
        candidates = sorted(input_dir.glob(f"*{sample_id}*_spot.csv"))
        if candidates:
            return str(candidates[0]), search_dirs
    return None, search_dirs


def run_best_builder(
    cfg: PipelineConfig,
    *,
    sample_id: str = "",
    scores_matrix: str = "",
    labels_matrix: str = "",
    spot_template: str = "",
    visium_dir: str = "",
    output_dir: str = "",
    smooth_knn: bool | None = None,
    truth_file: str = "",
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> Dict[str, Any]:
    """Execute ``ensemble/build_best.py``."""
    repo = cfg.repo_root()
    script = repo / "ensemble" / "build_best.py"
    if not script.exists():
        return {"ok": False, "error": f"Not found: {script}"}

    sid = sample_id or cfg.sample_id
    od = output_dir or str(cfg.resolved_best_output_dir())
    if hasattr(cfg, "resolved_scoring_consensus_dir"):
        consensus_dir = cfg.resolved_scoring_consensus_dir()
    else:
        legacy_consensus_dir = repo / "scoring" / "output" / "consensus"
        sample_consensus_dir = repo / "scoring" / "output" / sid / "consensus"
        consensus_dir = sample_consensus_dir if sample_consensus_dir.exists() else legacy_consensus_dir
    sm = scores_matrix or str(consensus_dir / "scores_matrix.csv")
    lm = labels_matrix or str(consensus_dir / "labels_matrix.csv")
    vd = visium_dir or cfg.data_path
    tf = truth_file or cfg.best_truth_file
    sk = smooth_knn if smooth_knn is not None else cfg.best_smooth_knn

    if not sid:
        return {"ok": False, "error": "sample_id is required"}

    sm_path = Path(sm)
    if not sm_path.exists():
        return {"ok": False, "error": f"scores_matrix not found: {sm_path}"}

    lm_path = Path(lm)
    if not lm_path.exists():
        return {"ok": False, "error": f"labels_matrix not found: {lm_path}"}

    # Auto-detect spot_template if not provided
    st = spot_template
    if not st:
        resolved_template, searched_dirs = _resolve_spot_template(cfg, sample_id=sid, repo=repo)
        if resolved_template:
            st = resolved_template
        else:
            searched = ", ".join(str(p) for p in searched_dirs) or "(none)"
            return {
                "ok": False,
                "error": (
                    f"No spot template found for sample_id={sid}. "
                    f"Searched directories: {searched}. Expected pattern: *{sid}*_spot.csv"
                ),
            }
    elif not Path(st).exists():
        return {"ok": False, "error": f"spot_template not found: {st}"}

    cmd = [
        sys.executable, str(script),
        "--sample_id", str(sid),
        "--scores_matrix", str(sm),
        "--labels_matrix", str(lm),
        "--spot_template", str(st),
        "--output_dir", str(od),
    ]
    if vd:
        cmd += ["--visium_dir", str(vd)]
    if sk:
        cmd.append("--smooth_knn")
    if tf:
        cmd += ["--truth_file", str(tf)]

    print(f"[Tool] Running BEST builder -> {od}")
    run_result = _run_command(
        cmd=cmd,
        cwd=repo,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    exit_code = int(run_result.get("returncode", 1))
    interrupted = bool(run_result.get("interrupted", False))
    return {
        "ok": (exit_code == 0 and not interrupted),
        "exit_code": exit_code,
        "interrupted": interrupted,
        "output_dir": od,
        "scores_matrix_used": str(sm_path),
        "labels_matrix_used": str(lm_path),
        "spot_template_used": str(st),
        "log_tail": run_result.get("stdout_tail", []),
        "log_line_count": int(run_result.get("log_line_count", 0)),
    }
