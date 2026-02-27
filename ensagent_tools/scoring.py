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
    effective_vlm_off = cfg.vlm_off if vlm_off is None else bool(vlm_off)
    if effective_vlm_off:
        cmd.append("--vlm_off")

    print(f"[Tool] Running scoring")
    p = subprocess.run(cmd, cwd=str(script.parent), check=False)
    return {"ok": p.returncode == 0, "exit_code": p.returncode}
