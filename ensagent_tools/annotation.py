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
    if domain:
        cmd += ["--domain", str(domain)]

    print(f"[Tool] Running multi-agent annotation -> {dd}")
    p = subprocess.run(cmd, cwd=str(script.parent), check=False)
    return {"ok": p.returncode == 0, "exit_code": p.returncode, "data_dir": dd}
