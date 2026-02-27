"""
End-to-end runner for EnsAgent (Tool-runner -> Scoring -> BEST -> Annotation).

Configuration priority (highest wins):
  1. CLI arguments           ``python endtoend.py --n_clusters 5``
  2. pipeline_config.yaml    ``pipeline_config.yaml`` in the repo root
  3. Built-in defaults       defined in ``PipelineConfig``

Typical usage:
  # First time -- copy the example and edit it
  cp pipeline_config.example.yaml pipeline_config.yaml
  # Then just run
  python endtoend.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ensagent_tools.config_manager import PipelineConfig, load_config
from ensagent_tools.pipeline import run_full_pipeline


def _build_cli_overrides(args: argparse.Namespace) -> dict:
    """Convert parsed CLI args into a dict, keeping only explicitly-set values."""
    overrides: dict = {}
    mapping = {
        "data_path": "data_path",
        "sample_id": "sample_id",
        "tool_output_dir": "tool_output_dir",
        "n_clusters": "n_clusters",
        "random_seed": "random_seed",
        "methods": "methods",
        "skip_tool_runner": "skip_tool_runner",
        "overwrite_staging": "overwrite_staging",
        "vlm_off": "vlm_off",
        "skip_scoring": "skip_scoring",
        "run_best": "run_best",
        "best_output_dir": "best_output_dir",
        "best_smooth_knn": "best_smooth_knn",
        "best_truth_file": "best_truth_file",
        "run_annotation_multiagent": "run_annotation_multiagent",
    }
    for cli_key, cfg_key in mapping.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            overrides[cfg_key] = val
    return overrides


def main() -> None:
    ap = argparse.ArgumentParser(
        description="EnsAgent end-to-end pipeline runner",
        epilog=(
            "If no arguments are provided, settings are loaded from "
            "pipeline_config.yaml in the repository root."
        ),
    )

    ap.add_argument("--config", default=None, help="Path to pipeline YAML config file (default: pipeline_config.yaml)")
    ap.add_argument("--data_path", default=None, help="Visium data directory")
    ap.add_argument("--sample_id", default=None, help="Sample identifier")
    ap.add_argument("--tool_output_dir", default=None, help="Tool-runner output directory")
    ap.add_argument("--n_clusters", type=int, default=None)
    ap.add_argument("--random_seed", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--skip_tool_runner", action="store_true", default=None)
    ap.add_argument("--overwrite_staging", action="store_true", default=None)
    ap.add_argument("--vlm_off", action="store_true", default=None, help="Disable visual-score integration in scoring")
    ap.add_argument("--skip_scoring", action="store_true", default=None)
    ap.add_argument("--run_best", action="store_true", default=None)
    ap.add_argument("--best_output_dir", default=None)
    ap.add_argument("--best_smooth_knn", action="store_true", default=None)
    ap.add_argument("--best_truth_file", default=None)
    ap.add_argument("--run_annotation_multiagent", action="store_true", default=None)

    args = ap.parse_args()

    config_path = args.config
    cli_overrides = _build_cli_overrides(args)

    cfg = load_config(path=config_path, cli_overrides=cli_overrides)

    if not cfg.data_path or not cfg.sample_id:
        print(
            "[Error] data_path and sample_id are required.\n"
            "Set them in pipeline_config.yaml or pass via CLI.\n"
            "  cp pipeline_config.example.yaml pipeline_config.yaml\n"
            "  # edit pipeline_config.yaml, then:\n"
            "  python endtoend.py"
        )
        sys.exit(1)

    print(f"[Info] Pipeline config loaded")
    print(f"  data_path  : {cfg.data_path}")
    print(f"  sample_id  : {cfg.sample_id}")
    print(f"  conda_exe  : {cfg.conda_exe}")
    print(f"  phases     : tool_runner={'skip' if cfg.skip_tool_runner else 'run'}"
          f" | scoring={'skip' if cfg.skip_scoring else 'run'}"
          f" | best={'run' if cfg.run_best else 'skip'}"
          f" | annotation={'run' if cfg.run_annotation_multiagent else 'skip'}")

    result = run_full_pipeline(cfg)

    if result["ok"]:
        print("\n✓ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Pipeline failed.")
        for phase, info in result.get("phases", {}).items():
            if not info.get("ok", True):
                print(f"  Failed phase: {phase} -- {info.get('error', '')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
