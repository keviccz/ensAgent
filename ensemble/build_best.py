from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def _read_truth_file(path: Path) -> Dict[str, str]:
    """Read truth mapping from a 2-column tsv: <spot_id> <label>."""
    m: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            spot_id, label = parts[0], parts[1]
            m[str(spot_id)] = str(label)
    return m


def _encode_labels(labels: List[str]) -> List[int]:
    """Deterministic label -> int encoding for ARI."""
    uniq = sorted(set(labels))
    idx = {k: i for i, k in enumerate(uniq)}
    return [idx[x] for x in labels]


def _choose_best_labels(scores_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.Series:
    """
    Re-implement the notebook selection logic:
    - For each spot, pick the method with max score
    - Tie-breaker: choose the candidate method whose label is most frequent among tied candidates
    - Final tie-breaker: deterministic by column order (first)
    """
    # Ensure aligned indices/columns
    common_cols = [c for c in scores_df.columns if c in labels_df.columns]
    scores = scores_df[common_cols]
    labels = labels_df[common_cols].astype(int)

    out = []
    for i in range(len(scores)):
        row_scores = scores.iloc[i].values.astype(float)
        row_labels = labels.iloc[i].values.astype(int)

        max_score = np.nanmax(row_scores)
        # candidates: all methods that share the max
        cand_idx = np.where(row_scores == max_score)[0]
        if len(cand_idx) == 1:
            chosen = int(cand_idx[0])
        else:
            cand_labels = row_labels[cand_idx]
            uniq, counts = np.unique(cand_labels, return_counts=True)
            max_count = int(np.max(counts)) if len(counts) else 0
            # labels with highest frequency among candidates
            top_lbls = set(uniq[counts == max_count].tolist())
            if len(top_lbls) == 1:
                # pick the first candidate that has the top label
                chosen = int(cand_idx[np.argmax([1 if l in top_lbls else 0 for l in cand_labels])])
            else:
                # deterministic fallback: pick first candidate by column order
                chosen = int(cand_idx[0])
        out.append(int(row_labels[chosen]))

    return pd.Series(out, index=scores_df.index, name="spatial_domain").astype(int)


def _smooth_knn_majority(
    spots_df: pd.DataFrame,
    labels: pd.Series,
    *,
    k: int = 19,
    majority_threshold: float = 0.5,
) -> pd.Series:
    """Lightweight spatial smoothing using kNN majority vote on x/y."""
    if "x" not in spots_df.columns or "y" not in spots_df.columns:
        return labels
    try:
        from sklearn.neighbors import NearestNeighbors

        coords = spots_df[["x", "y"]].astype(float).values
        n = int(len(coords))
        if n <= 5:
            return labels

        k_eff = int(max(3, min(int(k), n - 1)))
        nn = NearestNeighbors(n_neighbors=k_eff + 1)
        nn.fit(coords)
        idx = nn.kneighbors(coords, return_distance=False)

        lab = labels.values.astype(int)
        out = lab.copy()
        for i in range(n):
            nbrs = idx[i, 1:]  # drop self
            nbr_labels = lab[nbrs]
            cur = lab[i]
            frac = float(np.mean(nbr_labels == cur)) if len(nbr_labels) else 1.0
            if frac > float(majority_threshold):
                out[i] = cur
            else:
                # majority among neighbors
                u, c = np.unique(nbr_labels, return_counts=True)
                out[i] = int(u[int(np.argmax(c))]) if len(u) else cur
        return pd.Series(out, index=labels.index, name=labels.name).astype(int)
    except Exception:
        return labels


def _load_spot_template(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize spot_id column
    if "spot_id" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "spot_id"})
        else:
            # assume first col is spot_id
            df = df.rename(columns={df.columns[0]: "spot_id"})
    df["spot_id"] = df["spot_id"].astype(str)
    return df


def _write_placeholder_pathways(out_path: Path, domain_ids: List[int]) -> None:
    rows = []
    for d in domain_ids:
        rows.append(
            {
                "Domain": int(d),
                "Term": "NA",
                "NES": 0.0,
                "NOM p-val": 1.0,
                "Lead_genes": "",
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _compute_degs_from_visium(
    *,
    visium_dir: Path,
    spot_to_domain: pd.Series,
    out_path: Path,
    top_n: int = 10,
    min_abs_logfc: float = 1.0,
    max_adj_p: float = 0.05,
) -> None:
    import scanpy as sc

    adata = sc.read_visium(str(visium_dir))
    adata.var_names_make_unique()

    # align spots
    common = [s for s in spot_to_domain.index if s in adata.obs_names]
    if len(common) == 0:
        raise RuntimeError("No overlapping spot_ids between BEST spot labels and Visium data")

    adata = adata[common].copy()
    adata.obs["best_domain"] = spot_to_domain.loc[common].astype(int).astype("category").values

    # Normalize / log1p
    if "count" not in adata.layers:
        adata.layers["count"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4, layer="count")
    sc.pp.log1p(adata)

    sc.tl.rank_genes_groups(
        adata,
        groupby="best_domain",
        layer="count",
        method="wilcoxon",
        use_raw=False,
        key_added="de_genes",
    )
    deg_df = sc.get.rank_genes_groups_df(adata, group=None, key="de_genes")
    deg_df = deg_df.rename(columns={"group": "domain"})

    # filter and keep top_n per domain
    if "pvals_adj" in deg_df.columns:
        deg_df = deg_df[deg_df["pvals_adj"] <= float(max_adj_p)]
    if "logfoldchanges" in deg_df.columns:
        deg_df = deg_df[deg_df["logfoldchanges"].abs() >= float(min_abs_logfc)]

    # ensure deterministic sorting
    sort_cols = [c for c in ["domain", "pvals_adj", "scores"] if c in deg_df.columns]
    if sort_cols:
        deg_df = deg_df.sort_values(sort_cols, ascending=[True] + [True] * (len(sort_cols) - 1))

    deg_df = deg_df.groupby("domain").head(int(top_n)).reset_index(drop=True)
    deg_df.to_csv(out_path, index=False)


def _plot_result_png(spots_df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    df = spots_df.copy()
    df["spatial_domain"] = df["spatial_domain"].astype(int)
    doms = sorted(df["spatial_domain"].unique().tolist())

    cmap = plt.get_cmap("tab20", max(1, len(doms)))
    plt.figure(figsize=(8, 8))
    for i, d in enumerate(doms):
        sub = df[df["spatial_domain"] == d]
        plt.scatter(sub["x"], sub["y"], s=8, c=[cmap(i)], label=str(d), alpha=0.85)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.legend(title="Domain", bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase C: build BEST_* files (spot/DEGs/PATHWAY) and result image")
    ap.add_argument("--sample_id", required=True, help='Sample id, e.g. "DLPFC_151507" or "151507"')
    ap.add_argument("--scores_matrix", type=str, required=True, help="Path to scores_matrix.csv")
    ap.add_argument("--labels_matrix", type=str, required=True, help="Path to labels_matrix.csv")
    ap.add_argument("--spot_template", type=str, required=True, help="A *_spot.csv file providing x/y/spot_id columns")
    ap.add_argument("--visium_dir", type=str, default=None, help="Visium directory to compute BEST DEGs (optional)")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for BEST_* files")
    ap.add_argument("--smooth_knn", action="store_true", help="Enable kNN majority smoothing on BEST labels")
    ap.add_argument("--smooth_k", type=int, default=19)
    ap.add_argument("--smooth_threshold", type=float, default=0.5)
    ap.add_argument("--top_n_deg", type=int, default=10)
    ap.add_argument("--truth_file", type=str, default=None, help="Optional truth file (spot_id<TAB>label) to compute ARI")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_df = pd.read_csv(args.scores_matrix, index_col=0)
    labels_df = pd.read_csv(args.labels_matrix, index_col=0)
    labels_df = labels_df.astype(int)

    # align by spot_id
    common_idx = scores_df.index.intersection(labels_df.index)
    if len(common_idx) == 0:
        raise SystemExit("scores_matrix and labels_matrix have no overlapping spot_id index")
    scores_df = scores_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]

    template = _load_spot_template(Path(args.spot_template))
    template = template.set_index("spot_id", drop=False)
    common_idx = common_idx.intersection(template.index)
    if len(common_idx) == 0:
        raise SystemExit("No overlapping spot_id between matrices and spot_template")
    scores_df = scores_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]
    template = template.loc[common_idx]

    best_labels = _choose_best_labels(scores_df, labels_df)
    if args.smooth_knn:
        best_labels = _smooth_knn_majority(
            template,
            best_labels,
            k=int(args.smooth_k),
            majority_threshold=float(args.smooth_threshold),
        )

    best_spot = template.copy()
    best_spot["spatial_domain"] = best_labels.astype(int).values

    spot_out = out_dir / f"BEST_{args.sample_id}_spot.csv"
    best_spot.reset_index(drop=True).to_csv(spot_out, index=False)

    # BEST DEGs (optional but recommended)
    deg_out = out_dir / f"BEST_{args.sample_id}_DEGs.csv"
    if args.visium_dir:
        _compute_degs_from_visium(
            visium_dir=Path(args.visium_dir),
            spot_to_domain=best_labels,
            out_path=deg_out,
            top_n=int(args.top_n_deg),
        )
    else:
        # Write an empty DEG file with expected columns
        pd.DataFrame(columns=["domain", "names", "logfoldchanges", "pvals_adj", "scores"]).to_csv(deg_out, index=False)

    # PATHWAY placeholder (offline-safe)
    pw_out = out_dir / f"BEST_{args.sample_id}_PATHWAY.csv"
    domain_ids = sorted(set(int(x) for x in best_labels.unique()))
    _write_placeholder_pathways(pw_out, domain_ids)

    # Image for VLM
    img_out = out_dir / f"{args.sample_id}_result.png"
    _plot_result_png(best_spot, out_path=img_out, title=f"Spatial Clustering - {args.sample_id}")

    # Optional ARI
    if args.truth_file:
        truth = _read_truth_file(Path(args.truth_file))
        pred = best_labels.to_dict()
        overlap = sorted(set(truth.keys()) & set(pred.keys()))
        if overlap:
            y_true = [truth[s] for s in overlap]
            y_pred = [str(pred[s]) for s in overlap]
            ari = float(adjusted_rand_score(_encode_labels(y_true), _encode_labels(y_pred)))
            with (out_dir / "ari.json").open("w", encoding="utf-8") as f:
                json.dump({"sample_id": args.sample_id, "ari": ari, "n": len(overlap)}, f, indent=2)
            print(f"[Info] ARI={ari:.4f} (n={len(overlap)}) -> {out_dir/'ari.json'}")
        else:
            print("[Warning] truth_file provided but no overlapping spot_ids; ARI skipped")

    print(f"[Success] BEST artifacts written to: {out_dir}")
    print(f"  - {spot_out.name}")
    print(f"  - {deg_out.name}")
    print(f"  - {pw_out.name}")
    print(f"  - {img_out.name}")


if __name__ == "__main__":
    main()

