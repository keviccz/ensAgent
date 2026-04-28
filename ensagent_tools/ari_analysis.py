"""
Tool: compute_ari — ARI calculation + spatial clustering plot.

Converts ARI&picture_DLPFC_151507.ipynb into a callable tool.
All paths are auto-resolved from PipelineConfig; any can be overridden.
Returns ARI score, text report, and base64-encoded PNG for chat rendering.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _auto_paths(cfg) -> dict:
    """Resolve default paths from PipelineConfig."""
    repo = cfg.repo_root()
    sample_id = str(cfg.sample_id or "")
    return {
        "sample_id":    sample_id,
        "scores_dir":   str(repo / "scoring" / "output" / "consensus"),
        "visium_dir":   str(getattr(cfg, "data_path", "") or ""),
        "output_dir":   str(repo / "output" / "ari" / sample_id),
        "truth_file":   "",   # will look for metadata.tsv inside visium_dir
    }


def _find_truth_file(visium_dir: str, sample_id: str) -> Optional[str]:
    """Look for metadata.tsv / truth txt in common locations."""
    candidates = [
        Path(visium_dir) / "metadata.tsv",
        Path(visium_dir).parent / "metadata.tsv",
        Path(visium_dir) / f"{sample_id}_truth.txt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def compute_ari(
    cfg,
    *,
    sample_id: str = "",
    scores_dir: str = "",
    visium_dir: str = "",
    output_dir: str = "",
    truth_file: str = "",
    apply_smoothing: bool = True,
    n_clusters: int = 0,
) -> Dict[str, Any]:
    """Compute ARI and generate spatial clustering plot.

    Parameters
    ----------
    cfg:            PipelineConfig (auto-fills all path defaults)
    sample_id:      Override sample identifier.
    scores_dir:     Directory with scores_matrix.csv + labels_matrix.csv.
    visium_dir:     Visium data directory (for loading spatial coordinates).
    output_dir:     Where to save the PNG.
    truth_file:     Path to metadata.tsv / ground-truth labels.
    apply_smoothing: Whether to apply multi-scale spatial smoothing.
    n_clusters:     Expected cluster count (0 = auto-detect from data).
    """
    try:
        import numpy as np
        import pandas as pd
        import scanpy as sc
        import matplotlib
        matplotlib.use("Agg")   # headless
        import matplotlib.pyplot as plt
    except ImportError as e:
        return {"ok": False, "error": f"Missing dependency: {e}"}

    # ── 1. Resolve paths ────────────────────────────────────────────────────
    defaults = _auto_paths(cfg)
    sample_id  = sample_id  or defaults["sample_id"]
    scores_dir = scores_dir or defaults["scores_dir"]
    visium_dir = visium_dir or defaults["visium_dir"]
    output_dir = output_dir or defaults["output_dir"]
    truth_file = truth_file or defaults["truth_file"] or _find_truth_file(visium_dir, sample_id) or ""

    scores_path = Path(scores_dir) / "scores_matrix.csv"
    labels_path = Path(scores_dir) / "labels_matrix.csv"

    for p, name in [(scores_path, "scores_matrix.csv"), (labels_path, "labels_matrix.csv")]:
        if not p.exists():
            return {"ok": False, "error": f"Not found: {p}\nRun Stage B (run_scoring) first."}
    if not visium_dir or not Path(visium_dir).exists():
        return {"ok": False, "error": f"Visium directory not found: {visium_dir!r}\nSet data_path in pipeline_config.yaml."}

    os.makedirs(output_dir, exist_ok=True)

    # ── 2. Load Visium data ──────────────────────────────────────────────────
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = sc.read_visium(visium_dir)
        adata.var_names_make_unique()
    except Exception as e:
        return {"ok": False, "error": f"Failed to load Visium data from {visium_dir}: {e}"}

    # ── 3. Load ground-truth labels (optional) ───────────────────────────────
    has_truth = False
    if truth_file and Path(truth_file).exists():
        try:
            df_meta = pd.read_csv(truth_file, sep="\t")
            truth_col = next(
                (c for c in ["layer_guess", "label", "ground_truth", "truth"] if c in df_meta.columns),
                df_meta.columns[-1],
            )
            adata.obs["layer_guess"] = df_meta[truth_col].values[:len(adata.obs)]
            has_truth = True
        except Exception:
            pass

    # ── 4. Build ensemble labels from scores + label matrices ────────────────
    scores_df = pd.read_csv(scores_path, index_col=0)
    labels_df = pd.read_csv(labels_path, index_col=0).astype(int)

    min_rows = min(len(scores_df), len(labels_df))
    scores_df = scores_df.iloc[:min_rows]
    labels_df = labels_df.iloc[:min_rows]

    score2 = np.zeros(min_rows, dtype=int)
    for i in range(min_rows):
        score_row = scores_df.iloc[i].values
        label_row = labels_df.iloc[i].values
        max_score = np.max(score_row)
        max_idx = np.where(score_row == max_score)[0]
        if len(max_idx) == 1:
            selected = max_idx[0]
        else:
            cand_labels = label_row[max_idx]
            uq, cnt = np.unique(cand_labels, return_counts=True)
            if np.sum(cnt == cnt.max()) == 1:
                selected = max_idx[np.argmax(cnt)]
            else:
                selected = 2
        score2[i] = label_row[selected]

    result_df = pd.DataFrame({"spot_id": scores_df.index, "Ours_domain": score2}).set_index("spot_id")
    merged = adata.obs.merge(result_df, left_index=True, right_index=True, how="inner")
    adata = adata[merged.index, :].copy()
    adata.obs = merged

    # ── 5. Optional spatial smoothing ────────────────────────────────────────
    pixel_class = adata.obs["Ours_domain"].values.copy()

    if apply_smoothing:
        try:
            from scipy.spatial.distance import cdist
            coords = np.array(adata.obsm["spatial"], dtype=float)

            def _smooth_once(labels, k=19):
                result = np.zeros_like(labels)
                for i in range(len(coords)):
                    dists = cdist(coords[i:i+1], coords).flatten()
                    nbr = np.argsort(dists)[:k]
                    nbr_cls = labels[nbr]
                    cur = labels[i]
                    if np.mean(nbr_cls == cur) > 0.5:
                        result[i] = cur
                    else:
                        counts = np.bincount(nbr_cls.astype(int))
                        result[i] = int(np.argmax(counts))
                return result

            smoothed = _smooth_once(pixel_class.astype(int), k=19)
            smoothed = _smooth_once(smoothed, k=8)
        except Exception:
            smoothed = pixel_class.copy()
    else:
        smoothed = pixel_class.copy()

    adata.obs["smoothed_domain"] = pd.Categorical(smoothed)
    adata.obs["Ours_domain"] = pd.Categorical(pixel_class)

    # ── 6. ARI calculation ────────────────────────────────────────────────────
    ari_value = None
    ari_str = "N/A (no ground truth)"
    if has_truth:
        try:
            from sklearn import metrics
            ari_value = metrics.adjusted_rand_score(
                adata.obs["smoothed_domain"].astype("category").cat.codes,
                adata.obs["layer_guess"].astype("category").cat.codes,
            )
            ari_str = f"{ari_value:.4f}"
        except Exception as e:
            ari_str = f"error: {e}"

    # ── 7. Spatial plot ──────────────────────────────────────────────────────
    color_cols = []
    titles = []
    if has_truth:
        color_cols.append("layer_guess")
        titles.append("Ground Truth")
    color_cols.append("smoothed_domain")
    titles.append(f"EnsAgent Domains (ARI={ari_str})")
    color_cols.append("Ours_domain")
    titles.append("Raw Ensemble")

    try:
        sc.pl.spatial(
            adata,
            img_key="hires",
            color=color_cols,
            title=titles,
            alpha=0.85,
            spot_size=100,
            show=False,
        )
        fig = plt.gcf()
        fig.suptitle(f"Spatial Clustering — {sample_id}", fontsize=13, y=1.01)
        plt.tight_layout()

        # Save to file
        out_png = Path(output_dir) / f"{sample_id}_ari_plot.png"
        fig.savefig(str(out_png), bbox_inches="tight", dpi=120)

        # Encode to base64 for inline chat rendering
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close("all")
    except Exception as e:
        plt.close("all")
        return {"ok": False, "error": f"Plot failed: {e}"}

    # ── 8. Domain distribution report ────────────────────────────────────────
    dist = adata.obs["smoothed_domain"].value_counts().sort_index()
    dist_lines = "\n".join(f"  Domain {d}: {n} spots" for d, n in dist.items())
    report = (
        f"Sample: {sample_id}\n"
        f"Total spots: {len(adata)}\n"
        f"ARI vs ground truth: {ari_str}\n"
        f"Smoothing: {'enabled' if apply_smoothing else 'disabled'}\n"
        f"\nDomain distribution:\n{dist_lines}\n"
        f"\nPlot saved: {out_png}"
    )

    return {
        "ok": True,
        "sample_id": sample_id,
        "ari": ari_value,
        "ari_str": ari_str,
        "report": report,
        "image_path": str(out_png),
        "image_b64": img_b64,       # base64 PNG — frontend renders inline
        "n_spots": int(len(adata)),
        "n_domains": int(adata.obs["smoothed_domain"].nunique()),
    }
