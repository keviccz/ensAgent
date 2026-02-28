"""
Tool: run the full end-to-end pipeline (Stages A → B → C → D).

This is the core logic previously embedded in ``endtoend.py``, refactored
so it can be called programmatically *or* from the CLI.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict

from ensagent_tools.config_manager import PipelineConfig
from ensagent_tools.tool_runner import run_tool_runner
from ensagent_tools.scoring import run_scoring
from ensagent_tools.best_builder import run_best_builder
from ensagent_tools.annotation import run_annotation


_DOMAIN_PREFIX_RE = re.compile(r"^(?P<method>[^_]+)_domain_(?P<rest>.+)$")


def _rewrite_prefix(prefix: str) -> str:
    m = _DOMAIN_PREFIX_RE.match(prefix)
    if not m:
        return prefix
    return f"{m.group('method')}_{m.group('rest')}"


def stage_toolrunner_outputs(
    *,
    tool_output_dir: Path,
    scoring_input_dir: Path,
    sample_id: str,
    overwrite: bool,
) -> Dict[str, int]:
    """Copy Tool-runner outputs into ``scoring/input/`` with expected filenames."""
    spot_dir = tool_output_dir / "spot"
    deg_dir = tool_output_dir / "DEGs"
    pathway_dir = tool_output_dir / "PATHWAY"

    for d, label in [(spot_dir, "spot"), (deg_dir, "DEGs"), (pathway_dir, "PATHWAY")]:
        if not d.exists():
            raise FileNotFoundError(f"Not found: {d}")

    scoring_input_dir.mkdir(parents=True, exist_ok=True)

    def copy_group(src_dir: Path, suffix: str) -> int:
        n = 0
        for src in sorted(src_dir.glob(f"*_{suffix}.csv")):
            prefix = src.name[: -(len(f"_{suffix}.csv"))]
            if sample_id not in prefix:
                continue
            new_prefix = _rewrite_prefix(prefix)
            dst = scoring_input_dir / f"{new_prefix}_{suffix}.csv"
            if dst.exists() and not overwrite:
                continue
            shutil.copy2(src, dst)
            n += 1
        return n

    counts = {
        "spot": copy_group(spot_dir, "spot"),
        "DEGs": copy_group(deg_dir, "DEGs"),
        "PATHWAY": copy_group(pathway_dir, "PATHWAY"),
    }

    if counts["spot"] == 0:
        raise RuntimeError(f"No spot files staged from {spot_dir}")
    if counts["DEGs"] == 0:
        raise RuntimeError(f"No DEG files staged from {deg_dir}")
    if counts["PATHWAY"] == 0:
        print(f"[Warning] No PATHWAY files staged from {pathway_dir}")

    print(f"[Info] Staged files -> {scoring_input_dir}")
    for k, v in counts.items():
        print(f"  - {k}: {v}")
    return counts


def run_full_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    """Execute the complete A → B → C → D pipeline based on *cfg*."""
    repo = cfg.repo_root()
    results: Dict[str, Any] = {"ok": True, "phases": {}}

    tool_out = cfg.resolved_tool_output_dir()
    scoring_input = repo / "scoring" / "input"
    best_out = cfg.resolved_best_output_dir()

    # --- Stage A: Tool-runner ---
    if not cfg.skip_tool_runner:
        res = run_tool_runner(cfg, output_dir=str(tool_out))
        results["phases"]["tool_runner"] = res
        if not res["ok"]:
            results["ok"] = False
            return results

    # --- Stage B: Scoring ---
    if not cfg.skip_scoring:
        # --- Staging (only required before scoring) ---
        try:
            stage_toolrunner_outputs(
                tool_output_dir=tool_out,
                scoring_input_dir=scoring_input,
                sample_id=cfg.sample_id,
                overwrite=cfg.overwrite_staging,
            )
            results["phases"]["staging"] = {"ok": True}
        except Exception as e:
            results["phases"]["staging"] = {"ok": False, "error": str(e)}
            results["ok"] = False
            return results

        res = run_scoring(cfg)
        results["phases"]["scoring"] = res
        if not res["ok"]:
            results["ok"] = False
            return results
    else:
        results["phases"]["staging"] = {"ok": True, "skipped": True, "reason": "skip_scoring=true"}
        results["phases"]["scoring"] = {"ok": True, "skipped": True}

    # --- Stage C: BEST builder ---
    if cfg.run_best:
        res = run_best_builder(cfg, output_dir=str(best_out))
        results["phases"]["best_builder"] = res
        if not res["ok"]:
            results["ok"] = False
            return results

    # --- Stage D: Multi-agent annotation ---
    if cfg.run_annotation_multiagent:
        res = run_annotation(cfg, data_dir=str(best_out))
        results["phases"]["annotation"] = res
        if not res["ok"]:
            results["ok"] = False
            return results

    return results
