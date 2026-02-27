"""
Auto-discovery for Visium data path and sample id.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from streamlit_app.utils.config_bridge import get_repo_root


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _is_visium_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    spatial_ok = (path / "spatial").is_dir()
    matrix_h5_ok = (path / "filtered_feature_bc_matrix.h5").exists()
    matrix_dir_ok = (path / "filtered_feature_bc_matrix").is_dir()
    return spatial_ok and (matrix_h5_ok or matrix_dir_ok)


def _safe_yaml_load(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _sample_from_path(data_path: Path) -> str:
    return data_path.name


def _normalize_result(
    *,
    data_path: Path,
    sample_id: str,
    source: str,
    source_detail: str,
) -> Dict[str, Any]:
    return {
        "data_path": str(data_path),
        "sample_id": sample_id.strip() if sample_id else _sample_from_path(data_path),
        "source": source,
        "source_detail": source_detail,
        "warnings": [],
    }


def _discover_from_pipeline_config(repo_root: Path) -> Dict[str, Any] | None:
    cfg_path = repo_root / "pipeline_config.yaml"
    if not cfg_path.exists():
        return None
    raw = _safe_yaml_load(cfg_path)
    data_path_raw = str(raw.get("data_path") or "").strip()
    if not data_path_raw:
        return None

    data_path = _resolve_path(data_path_raw, repo_root)
    if not _is_visium_dir(data_path):
        return None

    sample_id = str(raw.get("sample_id") or "").strip()
    return _normalize_result(
        data_path=data_path,
        sample_id=sample_id,
        source="pipeline_config.yaml",
        source_detail="pipeline_config.yaml",
    )


def _discover_from_tool_runner_configs(repo_root: Path) -> Dict[str, Any] | None:
    cfg_dir = repo_root / "Tool-runner" / "configs"
    if not cfg_dir.exists():
        return None

    for cfg_path in sorted(cfg_dir.glob("*.yaml"), key=lambda p: p.name.lower()):
        raw = _safe_yaml_load(cfg_path)
        data_path_raw = str(raw.get("data_path") or "").strip()
        if not data_path_raw:
            continue
        candidates = [
            _resolve_path(data_path_raw, cfg_path.parent),
            _resolve_path(data_path_raw, repo_root),
        ]
        data_path = None
        for candidate in candidates:
            if _is_visium_dir(candidate):
                data_path = candidate
                break
        if data_path is None:
            continue

        sample_id = str(raw.get("sample_id") or "").strip()
        return _normalize_result(
            data_path=data_path,
            sample_id=sample_id,
            source="tool_runner_config",
            source_detail=str(cfg_path.resolve().as_posix()),
        )
    return None


def _iter_repo_visium_candidates(repo_root: Path) -> Iterable[Path]:
    for h5 in repo_root.rglob("filtered_feature_bc_matrix.h5"):
        candidate = h5.parent
        if _is_visium_dir(candidate):
            yield candidate.resolve()

    for matrix_dir in repo_root.rglob("filtered_feature_bc_matrix"):
        candidate = matrix_dir.parent
        if _is_visium_dir(candidate):
            yield candidate.resolve()


def _discover_from_repo_scan(repo_root: Path) -> Dict[str, Any] | None:
    candidates: List[Path] = sorted(
        set(_iter_repo_visium_candidates(repo_root)),
        key=lambda p: (len(p.parts), p.as_posix().lower()),
    )
    if not candidates:
        return None
    data_path = candidates[0]
    return _normalize_result(
        data_path=data_path,
        sample_id="",
        source="repo_scan",
        source_detail=data_path.as_posix(),
    )


def discover_data_defaults(repo_root: Path | str | None = None) -> Dict[str, Any]:
    """
    Discover best default Data Path and Sample ID.

    Priority:
      1) pipeline_config.yaml
      2) Tool-runner/configs/*.yaml
      3) repository scan for Visium directory markers
    """
    root = get_repo_root(repo_root).resolve()

    for finder in (
        _discover_from_pipeline_config,
        _discover_from_tool_runner_configs,
        _discover_from_repo_scan,
    ):
        result = finder(root)
        if result:
            return result

    return {
        "data_path": "",
        "sample_id": "",
        "source": "none",
        "source_detail": "",
        "warnings": ["No valid Visium directory found."],
    }
