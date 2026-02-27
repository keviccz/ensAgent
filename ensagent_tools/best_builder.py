"""
Tool: run the BEST builder (Stage C).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from ensagent_tools.config_manager import PipelineConfig


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
) -> Dict[str, Any]:
    """Execute ``ensemble/build_best.py``."""
    repo = cfg.repo_root()
    script = repo / "ensemble" / "build_best.py"
    if not script.exists():
        return {"ok": False, "error": f"Not found: {script}"}

    sid = sample_id or cfg.sample_id
    od = output_dir or str(cfg.resolved_best_output_dir())
    sm = scores_matrix or str(repo / "scoring" / "output" / "consensus" / "scores_matrix.csv")
    lm = labels_matrix or str(repo / "scoring" / "output" / "consensus" / "labels_matrix.csv")
    vd = visium_dir or cfg.data_path
    tf = truth_file or cfg.best_truth_file
    sk = smooth_knn if smooth_knn is not None else cfg.best_smooth_knn

    if not sid:
        return {"ok": False, "error": "sample_id is required"}

    # Auto-detect spot_template if not provided
    st = spot_template
    if not st:
        scoring_input = repo / "scoring" / "input"
        candidates = sorted(scoring_input.glob(f"*{sid}*_spot.csv"))
        if candidates:
            st = str(candidates[0])
        else:
            return {"ok": False, "error": f"No spot template found in {scoring_input} for {sid}"}

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
    p = subprocess.run(cmd, cwd=str(repo), check=False)
    return {"ok": p.returncode == 0, "exit_code": p.returncode, "output_dir": od}
