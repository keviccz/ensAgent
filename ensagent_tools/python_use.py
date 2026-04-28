"""
Tool: python_use — let the agent run a Python script or a code snippet.

Two modes:
  script  - run an existing .py file: python_use(script="scoring/pic_analyze/auto_analyzer.py", args=["--sample", "151507"])
  code    - execute inline code:       python_use(code="import os; print(os.listdir('.'))")

The subprocess always uses the ensagent conda environment's Python so that
all pipeline dependencies (scanpy, litellm, etc.) are available.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


# ── Resolve the ensagent Python executable ─────────────────────────────────────
def _find_python() -> str:
    candidates = [
        Path("E:/Miniforge3/envs/ensagent/python.exe"),
        Path("E:/Miniforge3/envs/ens_dev/python.exe"),
        Path(sys.executable),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


ENSAGENT_PYTHON = _find_python()
REPO_ROOT = Path(__file__).resolve().parent.parent


def python_use(
    cfg,
    *,
    script: str = "",
    code: str = "",
    module: str = "",
    args: List[str] | None = None,
    cwd: str = "",
    timeout: int = 120,
) -> Dict[str, Any]:
    """Execute a Python script, module, or inline code snippet.

    Parameters
    ----------
    cfg:     PipelineConfig (used for PYTHONPATH / repo root)
    script:  Path to a .py file (relative to repo root, or absolute).
    module:  Module path for package scripts with relative imports (e.g. 'scoring.pic_analyze.auto_analyzer').
    code:    Python source code to execute inline.
    args:    Extra CLI arguments.
    cwd:     Working directory (default: repo root).
    timeout: Max execution time in seconds (default: 120).
    """
    if not script and not code and not module:
        return {"ok": False, "error": "Provide 'script', 'module', or 'code'."}

    work_dir = Path(cwd) if cwd else REPO_ROOT

    # Build PYTHONPATH: repo root + working dir (for local bare imports like `from config import ...`)
    existing = os.environ.get("PYTHONPATH", "")
    py_paths = [str(REPO_ROOT), str(work_dir)]
    if existing:
        py_paths.append(existing)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(py_paths),
        "PYTHONIOENCODING": "utf-8",   # ensure emoji / CJK output works on Windows
    }

    tmp_file: str | None = None

    try:
        if module:
            # Resolve the .py file from the module path
            parts = module.split(".")
            script_path = REPO_ROOT.joinpath(*parts).with_suffix(".py")
            pkg_dir = script_path.parent  # e.g. scoring/pic_analyze/

            if script_path.exists():
                # Run as a direct script from its own directory so that:
                #   - sys.path[0] = pkg_dir  → bare `from config import Config` works
                #   - relative imports fall back to bare imports (try/except pattern)
                env["PYTHONPATH"] = (
                    str(REPO_ROOT) + os.pathsep + str(pkg_dir) + os.pathsep
                    + env.get("PYTHONPATH", "")
                )
                work_dir = pkg_dir  # override cwd to script's dir
                cmd = [ENSAGENT_PYTHON, str(script_path)] + (args or [])
            else:
                # Fallback: run as -m module
                cmd = [ENSAGENT_PYTHON, "-m", module] + (args or [])
        elif script:
            script_path = Path(script)
            if not script_path.is_absolute():
                script_path = REPO_ROOT / script
            if not script_path.exists():
                return {"ok": False, "error": f"Script not found: {script_path}"}
            cmd = [ENSAGENT_PYTHON, str(script_path)] + (args or [])
        else:
            # Write inline code to a temp file
            fd, tmp_file = tempfile.mkstemp(suffix=".py", prefix="ensagent_run_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)
            cmd = [ENSAGENT_PYTHON, tmp_file] + (args or [])

        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        exit_code = result.returncode

        # Trim very long output to keep chat readable
        MAX_CHARS = 4000
        if len(stdout) > MAX_CHARS:
            stdout = stdout[:MAX_CHARS] + f"\n… [truncated, {len(result.stdout)} chars total]"
        if len(stderr) > MAX_CHARS:
            stderr = stderr[:MAX_CHARS] + f"\n… [truncated, {len(result.stderr)} chars total]"

        return {
            "ok": exit_code == 0,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "summary": (stdout[:300] if stdout else stderr[:300]) or "(no output)",
        }

    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "exit_code": -1,
            "error": f"Script timed out after {timeout}s.",
            "stdout": "",
            "stderr": "",
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "stdout": "", "stderr": ""}
    finally:
        if tmp_file and Path(tmp_file).exists():
            try:
                os.unlink(tmp_file)
            except OSError:
                pass
