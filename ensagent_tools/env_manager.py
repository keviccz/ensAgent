"""
Conda / Mamba environment management tools.

Checks, creates, and validates R / PY / PY2 environments.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ensagent_tools.config_manager import PipelineConfig


def _conda(cfg: PipelineConfig) -> str:
    return cfg.conda_exe or "mamba"


def _env_yml_dir(cfg: PipelineConfig) -> Path:
    return cfg.repo_root() / "envs"


def check_envs(cfg: PipelineConfig) -> Dict[str, Any]:
    """Check whether required environments exist via ``mamba/conda env list``."""
    exe = _conda(cfg)
    out: Dict[str, Any] = {
        "conda_exe": exe,
        "env_names": cfg.env_names,
        "found": {},
        "missing": [],
        "warnings": [],
    }

    try:
        p = subprocess.run(
            [exe, "env", "list", "--json"],
            capture_output=True, text=True, check=False,
        )
        if p.returncode != 0:
            out["warnings"].append(f"{exe} env list failed: {p.stderr.strip()}")
            return {**out, "ok": False}
        data = json.loads(p.stdout)
        env_paths = data.get("envs", [])
    except FileNotFoundError:
        return {**out, "ok": False, "warnings": [f"Executable not found: {exe}"]}
    except Exception as e:
        return {**out, "ok": False, "warnings": [str(e)]}

    known_names = _list_env_names(env_paths)

    for label, name in cfg.env_names.items():
        if name in known_names:
            out["found"][label] = name
        else:
            out["missing"].append(label)

    out["ok"] = len(out["missing"]) == 0
    return out


def setup_envs(cfg: PipelineConfig) -> Dict[str, Any]:
    """Create missing environments from ``envs/*.yml``."""
    exe = _conda(cfg)
    yml_dir = _env_yml_dir(cfg)
    yml_map = {
        "R": yml_dir / "R_environment.yml",
        "PY": yml_dir / "PY_environment.yml",
        "PY2": yml_dir / "PY2_environment.yml",
    }

    status = check_envs(cfg)
    missing = status.get("missing", [])
    if not missing:
        return {"ok": True, "message": "All environments already exist", "created": []}

    created: List[str] = []
    for label in missing:
        yml = yml_map.get(label)
        if yml is None or not yml.exists():
            return {"ok": False, "error": f"Missing env yml for {label}: {yml}", "created": created}

        name = cfg.env_names.get(label, label)
        print(f"[Info] Creating environment '{name}' from {yml.name} (this may take a while) ...")
        p = subprocess.run(
            [exe, "env", "create", "-f", str(yml), "-n", name, "-y"],
            check=False,
        )
        if p.returncode != 0:
            return {"ok": False, "error": f"Failed to create env {name} (exit {p.returncode})", "created": created}
        created.append(name)

    return {"ok": True, "created": created}


def _list_env_names(env_paths: List[str]) -> set[str]:
    """Extract short env names from full prefix paths."""
    names: set[str] = set()
    for p in env_paths:
        names.add(Path(p).name)
    return names
