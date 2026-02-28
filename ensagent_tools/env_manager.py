"""
Conda / Mamba environment management tools.

Checks, creates, and validates R / PY / PY2 environments.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ensagent_tools.config_manager import PipelineConfig


def _conda(cfg: PipelineConfig) -> str:
    resolved = resolve_conda_executable(cfg)
    return resolved.get("exe") or cfg.conda_exe or "mamba"


def _normalize_candidate(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip().strip('"').strip("'")


def resolve_conda_executable(cfg: PipelineConfig) -> Dict[str, Any]:
    """Resolve a usable conda/mamba executable from config, env vars, then PATH."""
    configured = _normalize_candidate(cfg.conda_exe)
    env_mamba = _normalize_candidate(os.environ.get("MAMBA_EXE"))
    env_conda = _normalize_candidate(os.environ.get("CONDA_EXE"))

    candidates: List[tuple[str, str]] = []
    if configured:
        candidates.append((configured, "config"))
    if env_mamba:
        candidates.append((env_mamba, "env:MAMBA_EXE"))
    if env_conda:
        candidates.append((env_conda, "env:CONDA_EXE"))
    candidates.extend(
        [
            ("mamba", "path"),
            ("mamba.exe", "path"),
            ("mamba.bat", "path"),
            ("conda", "path"),
            ("conda.exe", "path"),
            ("conda.bat", "path"),
        ]
    )

    deduped: List[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate, source in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((candidate, source))

    checked: List[Dict[str, str]] = []
    for candidate, source in deduped:
        resolved = ""
        cpath = Path(candidate)
        if cpath.is_file():
            resolved = str(cpath)
        else:
            found = shutil.which(candidate)
            if found:
                resolved = found
        checked.append({"candidate": candidate, "source": source, "resolved": resolved})
        if resolved:
            return {
                "ok": True,
                "exe": resolved,
                "source": source,
                "requested": configured or cfg.conda_exe or "",
                "checked": checked,
            }

    return {
        "ok": False,
        "exe": "",
        "source": "",
        "requested": configured or cfg.conda_exe or "",
        "checked": checked,
    }


def _env_yml_dir(cfg: PipelineConfig) -> Path:
    return cfg.repo_root() / "envs"


def check_envs(cfg: PipelineConfig) -> Dict[str, Any]:
    """Check whether required environments exist via ``mamba/conda env list``."""
    resolved = resolve_conda_executable(cfg)
    exe = resolved.get("exe")
    out: Dict[str, Any] = {
        "requested_conda_exe": resolved.get("requested"),
        "conda_exe": exe or "",
        "conda_source": resolved.get("source", ""),
        "resolver_checked": resolved.get("checked", []),
        "env_names": cfg.env_names,
        "found": {},
        "missing": [],
        "warnings": [],
    }

    if not resolved.get("ok"):
        out["warnings"].append(
            "No usable conda/mamba executable found. "
            "Checked config, MAMBA_EXE/CONDA_EXE, and PATH."
        )
        return {**out, "ok": False}

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
    resolved = resolve_conda_executable(cfg)
    exe = resolved.get("exe")
    if not resolved.get("ok") or not exe:
        return {
            "ok": False,
            "error": (
                "No usable conda/mamba executable found. "
                "Checked config, MAMBA_EXE/CONDA_EXE, and PATH."
            ),
            "requested_conda_exe": resolved.get("requested"),
            "resolver_checked": resolved.get("checked", []),
        }

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
