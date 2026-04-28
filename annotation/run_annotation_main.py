#!/usr/bin/env python3
"""Standalone entry-point for Stage D multi-agent annotation.

Called by ensagent_tools/annotation.py to avoid the circular import that
occurs when launching through scoring/scoring.py.

Usage:
  python annotation/run_annotation_main.py \
    --data_dir output/best/151507 \
    --sample_id 151507 \
    [--api_provider azure] [--api_key ...] [--api_endpoint ...] \
    [--api_version ...] [--api_model ...] \
    [--domain 1,2,3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the repo root is importable (this file lives in annotation/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_config(args: argparse.Namespace):
    """Build a ScoreRagConfig with CLI credential overrides."""
    from scoring.config import ScoreRagConfig, load_config
    try:
        cfg = load_config()
    except Exception:
        cfg = ScoreRagConfig.from_env()

    updates: dict = {}
    if args.api_provider:
        updates["api_provider"] = args.api_provider
    if args.api_key:
        updates["api_key"] = args.api_key
    if args.api_endpoint:
        updates["api_endpoint"] = args.api_endpoint
    if args.api_version:
        updates["api_version"] = args.api_version
    if args.api_model:
        updates["api_model"] = args.api_model
    if updates:
        cfg = cfg.update(**updates)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="EnsAgent Stage D — multi-agent annotation")
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--sample_id",    required=True)
    parser.add_argument("--api_provider", default="")
    parser.add_argument("--api_key",      default="")
    parser.add_argument("--api_endpoint", default="")
    parser.add_argument("--api_version",  default="")
    parser.add_argument("--api_model",    default="")
    parser.add_argument("--domain",       default="",
                        help="Comma-separated domain IDs to annotate (default: all)")
    parser.add_argument("--output_dir",   default="")
    args = parser.parse_args()

    from annotation.annotation_multiagent.orchestrator import run_annotation_multiagent

    cfg = _build_config(args)

    target_domains = None
    if args.domain:
        try:
            target_domains = [int(x.strip()) for x in args.domain.split(",") if x.strip()]
        except ValueError:
            print(f"[Error] Invalid --domain value: {args.domain}", file=sys.stderr)
            return 1

    resolved_output_dir = Path(args.output_dir) if args.output_dir else Path(args.data_dir) / "annotation_output"

    print(f"[Info] Stage D: multi-agent annotation")
    print(f"[Info]   data_dir   : {args.data_dir}")
    print(f"[Info]   sample_id  : {args.sample_id}")
    print(f"[Info]   domains    : {target_domains or 'all'}")
    print(f"[Info]   provider   : {getattr(cfg, 'api_provider', '')}")

    results = run_annotation_multiagent(
        data_dir=args.data_dir,
        sample_id=args.sample_id,
        target_domains=target_domains,
        output_dir=str(resolved_output_dir),
        config=cfg,
    )

    out_path = resolved_output_dir / "domain_annotations.json"
    print(f"[Success] Annotation complete — {len(results)} domains")
    print(f"[Success] Output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
