from __future__ import annotations

import re
from pathlib import Path


def normalize_sample_slug(sample_id: str | None) -> str:
    value = str(sample_id or "").strip()
    if not value:
        return ""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return slug


def visual_scores_filename(sample_id: str | None) -> str:
    slug = normalize_sample_slug(sample_id)
    if not slug:
        return "all_domains_scores.json"
    return f"all_domains_scores_{slug}.json"


def visual_scores_path(pic_analyze_dir: str | Path, sample_id: str | None) -> Path:
    return Path(pic_analyze_dir) / "output" / visual_scores_filename(sample_id)
