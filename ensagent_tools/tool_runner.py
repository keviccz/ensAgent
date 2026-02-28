"""
Tool: run the Tool-runner clustering orchestrator (Stage A).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Dict, List, Optional

from ensagent_tools.config_manager import PipelineConfig
from ensagent_tools.env_manager import resolve_conda_executable


def run_tool_runner(
    cfg: PipelineConfig,
    *,
    data_path: str = "",
    sample_id: str = "",
    output_dir: str = "",
    n_clusters: int | None = None,
    random_seed: int | None = None,
    methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute ``Tool-runner/orchestrator.py`` with the given (or config-default) parameters."""
    repo = cfg.repo_root()
    script = repo / "Tool-runner" / "orchestrator.py"
    if not script.exists():
        return {"ok": False, "error": f"Not found: {script}"}

    dp = data_path or cfg.data_path
    sid = sample_id or cfg.sample_id
    od = output_dir or str(cfg.resolved_tool_output_dir())
    nc = n_clusters if n_clusters is not None else cfg.n_clusters
    rs = random_seed if random_seed is not None else cfg.random_seed
    meths = methods or cfg.methods

    if not dp or not sid:
        return {"ok": False, "error": "data_path and sample_id are required"}

    resolved_conda = resolve_conda_executable(cfg)
    conda_exe = resolved_conda.get("exe")
    if not resolved_conda.get("ok") or not conda_exe:
        return {
            "ok": False,
            "error": (
                "No usable conda/mamba executable found. "
                "Checked config, MAMBA_EXE/CONDA_EXE, and PATH."
            ),
            "requested_conda_exe": resolved_conda.get("requested"),
            "resolver_checked": resolved_conda.get("checked", []),
        }

    cmd = [
        sys.executable, str(script),
        "--data_path", str(dp),
        "--sample_id", str(sid),
        "--output_dir", str(od),
        "--n_clusters", str(int(nc)),
        "--random_seed", str(int(rs)),
        "--conda_exe", str(conda_exe),
    ]
    for label in ("R", "PY", "PY2"):
        cmd += [f"--env_{label.lower()}", cfg.env_names.get(label, label)]
    if meths:
        cmd += ["--methods", *meths]

    print(f"[Tool] Running Tool-runner -> {od}")
    p = subprocess.run(cmd, cwd=str(repo), check=False)
    return {
        "ok": p.returncode == 0,
        "exit_code": p.returncode,
        "output_dir": od,
        "conda_exe": str(conda_exe),
        "conda_source": resolved_conda.get("source", ""),
        "n_clusters_used": int(nc),
        "random_seed_used": int(rs),
        "methods_used": list(meths) if meths else [],
    }
