from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _load_pipeline_sample_id(repo_root: Path | None = None) -> str:
    if yaml is None:
        return ""
    root = repo_root or Path(__file__).resolve().parents[2]
    cfg_path = root / "pipeline_config.yaml"
    if not cfg_path.exists():
        return ""
    try:
        data: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return ""
    return str(data.get("sample_id") or "").strip()


def resolve_sample_id(
    sample_id: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    pipeline_sample_id: str | None = None,
) -> str:
    env_map = dict(os.environ) if env is None else dict(env)
    return _first_non_empty(
        sample_id,
        env_map.get("ENSAGENT_SAMPLE_ID", ""),
        pipeline_sample_id if pipeline_sample_id is not None else _load_pipeline_sample_id(),
    )


def normalize_sample_slug(sample_id: str | None) -> str:
    value = str(sample_id or "").strip()
    if not value:
        return ""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")


def display_sample_id(sample_id: str | None) -> str:
    return str(sample_id or "").strip() or "151507"


def visual_scores_filename(sample_id: str | None) -> str:
    slug = normalize_sample_slug(sample_id)
    if not slug:
        return "all_domains_scores.json"
    return f"all_domains_scores_{slug}.json"


def domain_report_filename(domain_num: int, sample_id: str | None) -> str:
    slug = normalize_sample_slug(sample_id)
    if not slug:
        return f"domain{int(domain_num)}_report.txt"
    return f"domain{int(domain_num)}_report_{slug}.txt"
