from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from scoring.config import ScoreRagConfig, load_config

from .critic import critic_agent
from .experts import marker_celltype_agent, pathway_agent, spatial_anatomy_agent, vlm_agent
from .llm_clients import make_provider_client, chat_json
from .logger import run_dir, write_json, utc_now_iso
from .proposer import compute_peer_scores, propose_annotation
from .schemas import (
    AllowedLabel,
    AnnotationSettings,
    CriticOutput,
    ExpertOutput,
    RoundLog,
)


DEFAULT_LABEL_SPACE: List[str] = [
    "Layer 1",
    "Layer 2",
    "Layer 3",
    "Layer 4",
    "Layer 5",
    "Layer 6",
    "White Matter",
    "Mixed L6/White Matter",
    "Mixed L1/L2",
    "Mixed L2/L3",
    "Mixed L3/L4",
    "Mixed L4/L5",
    "Mixed L5/L6",
]

def _repo_root_from_score_rag_dir() -> str:
    # This file lives in: scoring/annotation_multiagent/orchestrator.py
    # repo root is: <...>/scoring/.. (one level up)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _resolve_vlm_image_path(raw_path: str, *, data_dir: str, sample_id: str) -> str:
    """Resolve VLM image path robustly across different working directories.

    Order:
    1) If raw_path exists as-is -> return.
    2) If raw_path is relative -> try repo_root/raw_path.
    3) Try data_dir/<sample_id>_result.png (or basename).
    """
    if raw_path and os.path.exists(raw_path):
        return raw_path

    repo_root = _repo_root_from_score_rag_dir()
    if raw_path and not os.path.isabs(raw_path):
        cand = os.path.abspath(os.path.join(repo_root, raw_path))
        if os.path.exists(cand):
            return cand

    # try data_dir + basename
    base = os.path.basename(raw_path) if raw_path else f"{sample_id}_result.png"
    cand = os.path.join(data_dir, base)
    if os.path.exists(cand):
        return cand

    # return raw for error reporting
    return raw_path


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _compute_depth_model_pca(spots_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Compute a 2D PCA depth model (PC1) for the tissue coordinates.

    Returns a dict with keys: c0 (1x2), pc1 (2,), pmin, denom.
    If numpy is not available, returns None.
    """
    try:
        import numpy as np

        coords = spots_df[["x", "y"]].astype(float).values
        c0 = coords.mean(axis=0, keepdims=True)
        X = coords - c0
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        pc1 = vt[0]
        proj = (X @ pc1).astype(float)
        pmin, pmax = float(proj.min()), float(proj.max())
        denom = (pmax - pmin) if (pmax - pmin) > 0 else 1.0
        return {"c0": c0, "pc1": pc1, "pmin": pmin, "denom": denom, "method": "pca_pc1"}
    except Exception:
        return None


def _domain_depth_from_model(sub_df: pd.DataFrame, model: Dict[str, Any]) -> float:
    """Compute normalized depth (0..1) from a PCA model for a domain subset."""
    import numpy as np

    c0 = model["c0"]
    pc1 = model["pc1"]
    pmin = float(model["pmin"])
    denom = float(model["denom"]) if float(model["denom"]) > 0 else 1.0
    sub_coords = sub_df[["x", "y"]].astype(float).values
    sub_proj = ((sub_coords - c0) @ pc1).astype(float)
    return float((sub_proj.mean() - pmin) / denom)


def _compute_domain_depths(spots_df: pd.DataFrame, domain_ids: List[int], model: Optional[Dict[str, Any]]) -> Tuple[Dict[int, float], str]:
    """Precompute per-domain normalized depth using PCA model when possible, else y-based."""
    depths: Dict[int, float] = {}
    if model is not None:
        try:
            for d in domain_ids:
                sub = spots_df[spots_df["spatial_domain"] == d]
                if len(sub) == 0:
                    continue
                depths[int(d)] = _domain_depth_from_model(sub, model)
            return depths, str(model.get("method", "pca_pc1"))
        except Exception:
            # fall back to y-based
            depths = {}

    ymin, ymax = float(spots_df["y"].min()), float(spots_df["y"].max())
    denom = (ymax - ymin) if (ymax - ymin) > 0 else 1.0
    for d in domain_ids:
        sub = spots_df[spots_df["spatial_domain"] == d]
        if len(sub) == 0:
            continue
        depths[int(d)] = float((float(sub["y"].mean()) - ymin) / denom)
    return depths, "y_normalized"


def _build_domain_adjacency_knn(
    spots_df: pd.DataFrame,
    *,
    k: int = 8,
    min_edges: int = 30,
) -> Dict[int, Dict[int, float]]:
    """Build a domain adjacency graph using spot-level kNN.

    Returns adjacency weights per domain: adj[d][n] = weight (normalized counts).
    This is used for optional depth smoothing / layer-order consistency.
    """
    if "x" not in spots_df.columns or "y" not in spots_df.columns or "spatial_domain" not in spots_df.columns:
        return {}
    try:
        import numpy as np
        from sklearn.neighbors import NearestNeighbors

        coords = spots_df[["x", "y"]].astype(float).values
        dom = spots_df["spatial_domain"].astype(int).values
        n = int(len(dom))
        if n <= 10:
            return {}

        k_eff = int(max(3, min(int(k), 50)))
        nn = NearestNeighbors(n_neighbors=min(k_eff + 1, n), algorithm="auto")
        nn.fit(coords)
        _, idx = nn.kneighbors(coords, return_distance=True)

        # count cross-domain edges (undirected)
        pair_counts: Dict[Tuple[int, int], int] = {}
        for i in range(n):
            di = int(dom[i])
            # idx[i,0] is self
            for j in idx[i, 1:]:
                dj = int(dom[int(j)])
                if di == dj:
                    continue
                a, b = (di, dj) if di < dj else (dj, di)
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

        # expand to per-node adjacency counts
        adj_counts: Dict[int, Dict[int, int]] = {}
        for (a, b), c in pair_counts.items():
            if c < int(min_edges):
                continue
            adj_counts.setdefault(a, {})[b] = adj_counts.get(a, {}).get(b, 0) + int(c)
            adj_counts.setdefault(b, {})[a] = adj_counts.get(b, {}).get(a, 0) + int(c)

        # normalize per node
        adj: Dict[int, Dict[int, float]] = {}
        for d, nbrs in adj_counts.items():
            total = float(sum(nbrs.values())) or 1.0
            adj[d] = {n: float(c) / total for n, c in nbrs.items()}
        return adj
    except Exception:
        return {}


def _smooth_depths_by_adjacency(
    depths: Dict[int, float],
    adj: Dict[int, Dict[int, float]],
    *,
    alpha: float = 0.35,
    iters: int = 3,
) -> Dict[int, float]:
    """Laplacian-style smoothing on per-domain depths to encourage local consistency."""
    a = float(max(0.0, min(1.0, float(alpha))))
    t = int(max(1, min(20, int(iters))))
    cur = {int(k): float(v) for k, v in depths.items()}
    for _ in range(t):
        nxt = dict(cur)
        for d, dv in cur.items():
            nbrs = adj.get(int(d), {}) or {}
            if not nbrs:
                continue
            num = 0.0
            den = 0.0
            for n, w in nbrs.items():
                if int(n) not in cur:
                    continue
                ww = float(w)
                num += ww * float(cur[int(n)])
                den += ww
            if den > 0:
                avg = num / den
                nxt[int(d)] = float((1.0 - a) * float(dv) + a * float(avg))
        # clamp
        cur = {k: float(max(0.0, min(1.0, v))) for k, v in nxt.items()}
    return cur


def _surface_score_from_vlm(vlm: ExpertOutput) -> float:
    """Heuristic: how likely this domain is superficial (Layer 1/2)."""
    ls = vlm.get("label_support", {}) or {}
    try:
        return (
            1.0 * _safe_float(ls.get("Layer 1", 0.0))
            + 0.8 * _safe_float(ls.get("Mixed L1/L2", 0.0))
            + 0.5 * _safe_float(ls.get("Layer 2", 0.0))
        )
    except Exception:
        return 0.0


def _load_inputs(data_dir: str, sample_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Support both legacy naming and generalized naming:
    # - BEST_DLPFC_<sample_id>_*.csv (legacy)
    # - BEST_<sample_id>_*.csv (general)
    candidates = [
        (
            os.path.join(data_dir, f"BEST_{sample_id}_DEGs.csv"),
            os.path.join(data_dir, f"BEST_{sample_id}_PATHWAY.csv"),
            os.path.join(data_dir, f"BEST_{sample_id}_spot.csv"),
        ),
        (
            os.path.join(data_dir, f"BEST_DLPFC_{sample_id}_DEGs.csv"),
            os.path.join(data_dir, f"BEST_DLPFC_{sample_id}_PATHWAY.csv"),
            os.path.join(data_dir, f"BEST_DLPFC_{sample_id}_spot.csv"),
        ),
    ]
    degs_path = pathway_path = spot_path = ""
    for d, p, s in candidates:
        if os.path.exists(d) and os.path.exists(p) and os.path.exists(s):
            degs_path, pathway_path, spot_path = d, p, s
            break
    if not (degs_path and pathway_path and spot_path):
        tried = [f"{x[0]}, {x[1]}, {x[2]}" for x in candidates]
        raise FileNotFoundError(f"Missing BEST files for sample_id={sample_id} in {data_dir}. Tried: {tried}")
    degs_df = pd.read_csv(degs_path)
    pathways_df = pd.read_csv(pathway_path)
    spots_df = pd.read_csv(spot_path)
    return degs_df, pathways_df, spots_df


def _get_top_degs(degs_df: pd.DataFrame, domain_id: int, top_n: int = 15) -> List[Dict[str, Any]]:
    col_domain = "domain" if "domain" in degs_df.columns else ("group" if "group" in degs_df.columns else None)
    if col_domain is None or "names" not in degs_df.columns:
        return []
    df = degs_df[degs_df[col_domain] == domain_id].copy()
    if "logfoldchanges" in df.columns:
        df = df.sort_values("logfoldchanges", ascending=False)
        # Use upregulated DEGs only to avoid misleading "downregulation" claims in annotation.
        try:
            df_pos = df[df["logfoldchanges"] > 0]
            if len(df_pos) > 0:
                df = df_pos
        except Exception:
            # keep df as-is if column parsing fails
            pass
    # Filter out common non-informative genes to reduce hallucinations and improve interpretability
    def is_informative(g: str) -> bool:
        gu = g.upper()
        if gu.startswith("MT-"):
            return False
        if gu.startswith("RPL") or gu.startswith("RPS"):
            return False
        if gu in {"GAPDH", "ACTB", "B2M", "MALAT1"}:
            return False
        return True

    df["names"] = df["names"].astype(str)
    df_f = df[df["names"].apply(is_informative)]
    pool = (df_f if len(df_f) > 0 else df).copy()

    # --- Layer-specific marker injection ---
    # Many layer markers are not the highest logFC; ensure we include them if present.
    layer_markers = {
        # L1 / superficial
        "RELN",
        "CXCL14",
        "LHX2",
        "CPLX3",
        # L2/3 IT
        "CUX1",
        "CUX2",
        "CALB1",
        "SATB2",
        "RGS4",
        # L4 IT
        "RORB",
        "PCP4",
        "PDYN",
        # L5 ET/IT
        "BCL11B",
        "FEZF2",
        "TSHZ2",
        # L6 CT/IT
        "TLE4",
        "FOXP2",
        "CRYM",
        "TBR1",
        # WM / oligodendrocyte
        "MBP",
        "PLP1",
        "MOG",
        "MAG",
        "MOBP",
        "CNP",
        "CLDN11",
    }
    try:
        name_u = pool["names"].astype(str).str.upper()
        pool = pool.assign(_NAME_U=name_u)
    except Exception:
        pool = pool.assign(_NAME_U=pool["names"].astype(str))

    # take top-N by logFC, but prepend any layer markers found in the pool (up to 12)
    top_main = pool.head(int(top_n))
    try:
        extra = pool[pool["_NAME_U"].isin(layer_markers)].head(12)
    except Exception:
        extra = pool.head(0)
    combined = (
        pd.concat([extra, top_main], axis=0, ignore_index=True)
        .drop_duplicates(subset=["_NAME_U"], keep="first")
        .head(int(top_n))
    )

    out = []
    for _, row in combined.iterrows():
        out.append(
            {
                "gene": str(row.get("names", "")),
                "logfc": round(_safe_float(row.get("logfoldchanges", 0.0)), 3),
            }
        )
    return out


def _get_top_pathways(pathways_df: pd.DataFrame, domain_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
    if "Domain" not in pathways_df.columns or "Term" not in pathways_df.columns:
        return []
    df = pathways_df[pathways_df["Domain"] == domain_id].copy()
    if "NES" in df.columns:
        # Keep both enriched and depleted signals
        df = df.copy()
        df["abs_nes"] = df["NES"].abs()
        df = df.sort_values(["abs_nes", "NOM p-val"] if "NOM p-val" in df.columns else ["abs_nes"], ascending=[False, True])
    # Filter generic pathways that are often uninformative for layer identity
    generic_terms = {
        "Ribosome",
        "Coronavirus disease",
        "Oocyte meiosis",
    }
    try:
        df_f = df[~df["Term"].astype(str).isin(generic_terms)]
        top = (df_f if len(df_f) >= max(3, top_n) else df).head(top_n)
    except Exception:
        top = df.head(top_n)
    out = []
    for _, row in top.iterrows():
        out.append(
            {
                "term": str(row.get("Term", "")),
                "nes": round(_safe_float(row.get("NES", 0.0)), 3),
                "pval": round(_safe_float(row.get("NOM p-val", 1.0)), 6) if "NOM p-val" in row else None,
                "direction": "enriched" if _safe_float(row.get("NES", 0.0)) > 0 else "depleted",
            }
        )
    return out


def _compute_spatial_summary(
    spots_df: pd.DataFrame,
    domain_id: int,
    *,
    domain_depth: Optional[float] = None,
    domain_depth_raw: Optional[float] = None,
    adjacent_domains: Optional[List[Dict[str, Any]]] = None,
    depth_method: str = "y_normalized",
) -> Dict[str, Any]:
    # Expect columns: spatial_domain, x, y
    if "spatial_domain" not in spots_df.columns:
        return {"description": "Missing spatial_domain column", "num_spots": 0}
    if "x" not in spots_df.columns or "y" not in spots_df.columns:
        return {"description": "Missing x/y coordinates", "num_spots": 0}

    sub = spots_df[spots_df["spatial_domain"] == domain_id]
    n = int(len(sub))
    if n == 0:
        return {"description": "Domain not found in spots", "num_spots": 0}

    # Depth axis prior:
    # - If caller provides calibrated domain_depth, use it (0=superficial, 1=deep).
    # - Otherwise fall back to y-based normalization (stable and cheap).
    if domain_depth is None:
        ymin, ymax = float(spots_df["y"].min()), float(spots_df["y"].max())
        denom = (ymax - ymin) if (ymax - ymin) > 0 else 1.0
        dom_depth = float((float(sub["y"].mean()) - ymin) / denom)
        depth_method = "y_normalized"
    else:
        dom_depth = float(domain_depth)

    # Map depth to coarse regions with mixed bands
    # 0.00..0.15: L1/L2 (Mixed L1/L2)
    # 0.15..0.35: L2/L3 (Mixed L2/L3)
    # 0.35..0.70: mid/deep (L3/L5 prior weak)
    # 0.70..0.85: L5/L6 (Mixed L5/L6)
    # 0.85..1.00: deep/WM edge
    if dom_depth < 0.15:
        location = "Superficial (L1/L2 region)"
    elif dom_depth < 0.35:
        location = "Upper-mid (L2/L3 region)"
    elif dom_depth < 0.70:
        location = "Mid-depth (L3-L5 region)"
    elif dom_depth < 0.85:
        location = "Deep (L5/L6 region)"
    else:
        location = "Deepest / White-matter-adjacent region"

    # density heuristic: n / bbox_area as a simple proxy (no scipy dependency for hull)
    x_rng = float(sub["x"].max() - sub["x"].min())
    y_rng = float(sub["y"].max() - sub["y"].min())
    bbox_area = max(x_rng * y_rng, 1e-6)
    density = float(n / bbox_area)
    density_desc = "High Density" if density > 0.0005 else "Low Density/Dispersed"

    return {
        "num_spots": n,
        "location": location,
        "normalized_depth": round(float(dom_depth), 3),
        "normalized_depth_raw": (round(float(domain_depth_raw), 3) if domain_depth_raw is not None else None),
        "depth_method": depth_method,
        "density": density_desc,
        "bbox_area": round(bbox_area, 6),
        "density_value": round(density, 6),
        "adjacent_domains": (adjacent_domains or []),
    }


def _has_blocker(
    critic: CriticOutput,
    *,
    proposed_label: str = "",
    severity_threshold: float = 0.4,
) -> bool:
    """Whether critic issues should hard-block the run.

    Note: `white_matter_trap` is treated as a blocker ONLY when the proposed label is 'White Matter'.
    Otherwise, it is advisory (because the system is already avoiding the WM mislabel).
    """
    pl = str(proposed_label or "")
    for issue in critic.get("issues", []) or []:
        try:
            if str(issue.get("type", "")).strip() == "white_matter_trap" and pl != "White Matter":
                continue
            if bool(issue.get("blocker")) and _safe_float(issue.get("severity", 0.0)) >= severity_threshold:
                return True
        except Exception:
            continue
    return False


def _vlm_failed(vlm: ExpertOutput, vlm_min_score: float) -> bool:
    flags = set(vlm.get("quality_flags", []) or [])
    if {"vlm_required_failed", "vlm_unavailable", "vlm_parse_failed", "vlm_low_quality"} & flags:
        return True
    return _safe_float(vlm.get("agent_score", 0.0)) < float(vlm_min_score)


def _extract_suggested_labels(text: str, label_space: List[str]) -> List[str]:
    """Extract any label_space entries mentioned verbatim in free-text."""
    if not text:
        return []
    t = str(text)
    hits: List[str] = []
    for lbl in label_space:
        if lbl and lbl in t:
            hits.append(lbl)
    # keep order as they appear in label_space (deterministic).
    # Preference:
    # - Prefer Mixed labels when present.
    # - Do NOT auto-prioritize "White Matter" merely because it was mentioned in text.
    #   Critic often mentions "White Matter" in a NEGATIVE context (e.g., "White Matter is invalid"),
    #   which previously caused incorrect revise_only -> White Matter.
    uniq: List[str] = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    mixed = [h for h in uniq if str(h).startswith("Mixed ")]
    rest = [h for h in uniq if not str(h).startswith("Mixed ")]
    ordered = mixed + rest

    # Lightweight negation guard: drop labels that are mentioned only in a negative/avoid context.
    # This is intentionally conservative; auto-revision should rely mostly on explicit fix_hints.
    neg_words = [
        "not",
        "do not",
        "don't",
        "avoid",
        "reject",
        "rejected",
        "invalid",
        "no evidence",
        "lack",
        "lacks",
        "insufficient",
        "not supported",
        "cannot",
    ]
    cleaned: List[str] = []
    tl = t.lower()
    for lbl in ordered:
        try:
            pos = tl.find(str(lbl).lower())
            if pos < 0:
                continue
            window = tl[max(0, pos - 40) : min(len(tl), pos + 40)]
            if any(w in window for w in neg_words):
                continue
            cleaned.append(lbl)
        except Exception:
            cleaned.append(lbl)
    return cleaned


def _best_label_excluding(
    experts: Dict[str, ExpertOutput],
    label_space: List[str],
    weights: Dict[str, float],
    *,
    exclude: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """Compute best label from current expert label_support, excluding some labels."""
    exclude_set = set(exclude or [])
    # normalize over agents that provide label_support (Pathway excluded implicitly)
    label_weights: Dict[str, float] = {}
    for agent, out in experts.items():
        sup = out.get("label_support", {}) or {}
        if sup:
            label_weights[agent] = float(weights.get(agent, 0.0))
    s = sum(label_weights.values()) or 1.0
    ws = {k: float(v) / s for k, v in label_weights.items()}

    score_by_label: Dict[str, float] = {lbl: 0.0 for lbl in label_space}
    for agent, out in experts.items():
        sup = out.get("label_support", {}) or {}
        w = ws.get(agent, 0.0)
        for lbl, v in sup.items():
            if lbl not in score_by_label or lbl in exclude_set:
                continue
            try:
                score_by_label[lbl] += w * float(v)
            except Exception:
                continue

    ranked = sorted(score_by_label.items(), key=lambda t: float(t[1]), reverse=True)
    for lbl, sc in ranked:
        if lbl in exclude_set:
            continue
        return lbl, float(sc)
    return "Unknown", 0.0


def _label_index(lbl: str) -> Optional[float]:
    """Map labels to an ordered axis for disagreement checks (approximate cortical depth)."""
    if not lbl:
        return None
    s = str(lbl).strip()
    if s == "Mixed L6/White Matter":
        # boundary between deepest gray (L6) and WM
        return 6.75
    if s.startswith("Layer "):
        try:
            return float(int(s.replace("Layer ", "").strip()))
        except Exception:
            return None
    if s == "White Matter":
        return 7.0
    if s.startswith("Mixed "):
        # Examples: Mixed L2/L3, Mixed L3/L4, Mixed L5/L6, Mixed L1/L2
        try:
            part = s.replace("Mixed", "").strip()
            part = part.replace("L", "")
            a, b = part.split("/")
            return (float(a) + float(b)) / 2.0
        except Exception:
            return None
    return None


def _top_label(out: ExpertOutput) -> Tuple[str, float]:
    sup = out.get("label_support", {}) or {}
    best = ("Unknown", 0.0)
    for k, v in sup.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if fv > float(best[1]):
            best = (str(k), float(fv))
    return best


def _consensus_label_excluding_agent(
    experts: Dict[str, ExpertOutput],
    *,
    exclude_agent: str,
    label_space: List[str],
    weights: Dict[str, float],
) -> Tuple[str, float]:
    # Build a filtered view excluding the target agent.
    filtered: Dict[str, ExpertOutput] = {k: v for k, v in experts.items() if k != exclude_agent}
    return _best_label_excluding(filtered, label_space, weights, exclude=[])


def _weighted_consensus_from_top_votes(
    experts: Dict[str, ExpertOutput],
    *,
    exclude_agent: str,
    weights: Dict[str, float],
    min_vote_support: float = 0.55,
) -> Tuple[Optional[str], float, int, int, List[Dict[str, Any]]]:
    """Compute a robust consensus label based on top-1 votes from experts.

    Returns:
    - consensus_label (or None if no reliable consensus)
    - consensus_weight_ratio in [0,1] among considered voters
    - n_voters
    - voters: [{"agent":..., "weight":..., "label":..., "support":...}, ...]

    Consensus is only valid if:
    - at least 2 voters
    - weight_ratio is high (checked by caller)
    """
    voters: List[Dict[str, Any]] = []
    for agent, out in experts.items():
        if agent == exclude_agent:
            continue
        sup = out.get("label_support", {}) or {}
        if not sup:
            continue
        lbl, sc = _top_label(out)
        if not lbl:
            continue
        if float(sc) < float(min_vote_support):
            continue
        w = float(weights.get(agent, 0.0))
        if w <= 0:
            continue
        voters.append({"agent": agent, "weight": w, "label": lbl, "support": float(sc)})

    if len(voters) < 2:
        return None, 0.0, len(voters), 0, voters

    # Weighted vote by agent weights (not by label_support), because weights encode trust.
    by_label: Dict[str, float] = {}
    for v in voters:
        by_label[str(v["label"])] = by_label.get(str(v["label"]), 0.0) + float(v["weight"])
    ranked = sorted(by_label.items(), key=lambda t: float(t[1]), reverse=True)
    if not ranked:
        return None, 0.0, len(voters), 0, voters
    best_lbl, best_w = ranked[0]
    total_w = sum(float(v["weight"]) for v in voters) or 1.0
    ratio = float(best_w) / float(total_w)
    supporters = sum(1 for v in voters if str(v.get("label")) == str(best_lbl))
    return str(best_lbl), ratio, len(voters), int(supporters), voters


def _detect_disagreement_reruns(
    experts: Dict[str, ExpertOutput],
    *,
    label_space: List[str],
    weights: Dict[str, float],
) -> Tuple[List[str], Dict[str, str], List[str]]:
    """Detect strong disagreements and return (rerun_agents, guidance_by_agent, reason_parts)."""
    rerun: List[str] = []
    guidance: Dict[str, str] = {}
    reasons: List[str] = []

    for agent, out in experts.items():
        sup = out.get("label_support", {}) or {}
        if not sup:
            continue  # Pathway or missing agent doesn't participate in label disagreement

        a_lbl, a_sc = _top_label(out)
        if not a_lbl:
            continue

        # Robust consensus: must come from >=2 experts and have high weighted agreement.
        c_lbl, c_ratio, n_voters, n_supporters, voters = _weighted_consensus_from_top_votes(
            experts, exclude_agent=agent, weights=weights, min_vote_support=0.55
        )
        if not c_lbl:
            continue
        # "共识来自至少2个专家且一致性高":
        # - at least 2 experts vote for the SAME label
        # - and that label dominates the total (weight) vote.
        if int(n_supporters) < 2:
            continue
        if float(c_ratio) < 0.70 or int(n_voters) < 2:
            continue
        if a_lbl == c_lbl:
            continue

        ai = _label_index(a_lbl)
        ci = _label_index(c_lbl)
        # Require strong confidence and sufficiently large layer distance.
        # For Spatial, allow a smaller distance because depth thresholding can be off by ~1 layer.
        large_gap = False
        if ai is not None and ci is not None:
            gap = abs(float(ai) - float(ci))
            if str(agent) == "Spatial":
                large_gap = gap >= 1.0
            else:
                large_gap = gap >= 2.0
        else:
            # if non-ordinal labels (Unknown/Mixed), require very high confidence disagreement
            large_gap = False

        strong = float(a_sc) >= 0.60
        if strong and large_gap:
            # Trust consensus more when it is supported by >=2 experts AND dominated by higher total weight.
            # If the current agent itself has higher weight than the whole consensus ratio suggests, be conservative.
            agent_w = float(weights.get(agent, 0.0))
            total_w = sum(float(v["weight"]) for v in voters) or 1.0
            consensus_w = float(c_ratio) * float(total_w)
            if agent_w > consensus_w:
                # agent is more trusted than the consensus; don't force rerun based on weaker consensus.
                continue
            if agent not in rerun:
                rerun.append(agent)
            reasons.append(
                f"disagreement:{agent}({a_lbl},{a_sc:.2f})!=consensus({c_lbl},ratio={c_ratio:.2f},supporters={n_supporters})"
            )
            # Build a compact voter summary (top by weight)
            v_sorted = sorted(voters, key=lambda x: float(x.get("weight", 0.0)), reverse=True)[:4]
            voter_str = ", ".join([f"{v['agent']}({v['label']},{v['weight']:.2f})" for v in v_sorted])
            guidance[agent] = (
                "You previously disagreed with the peer consensus.\n"
                f"- Your top label: {a_lbl} (support={a_sc:.2f})\n"
                f"- Peer consensus (excluding you): {c_lbl} (weighted_agreement={c_ratio:.2f}, supporters={n_supporters})\n"
                f"- Consensus voters (by weight): {voter_str}\n"
                "Please re-evaluate your label_support and reasoning. If uncertainty remains, prefer Mixed labels near boundaries.\n"
            )

    return rerun, guidance, reasons


def _prefer_rerun_over_revise_for_conflict(
    *,
    experts: Dict[str, ExpertOutput],
    label_space: List[str],
    weights: Dict[str, float],
    existing_reruns: List[str],
    guidance_by_agent: Dict[str, str],
    issues: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, str], List[str]]:
    """Policy: for label conflicts, rerun the outlier expert with guidance rather than revise-only."""
    reason_parts: List[str] = []
    conflict_types = {"label_conflict", "label_confidence_low", "confidence_low"}
    has_conflict = False
    for it in issues or []:
        try:
            if str(it.get("type", "")).strip() in conflict_types:
                has_conflict = True
                break
        except Exception:
            continue
    if not has_conflict:
        return existing_reruns, guidance_by_agent, reason_parts

    rr = list(existing_reruns or [])

    # Use disagreement detector (robust consensus >=2 supporters, high weighted agreement)
    dis_reruns, dis_guidance, dis_reasons = _detect_disagreement_reruns(
        experts, label_space=label_space, weights=weights
    )
    for a in dis_reruns:
        if a not in rr:
            rr.append(a)
    if dis_guidance:
        guidance_by_agent.update(dis_guidance)
    if dis_reasons:
        reason_parts.append("conflict_outlier_rerun:" + ",".join(dis_reasons[:2]))

    # Special-case: surface-like consensus (Marker+Visual) vs Spatiality L2/L3
    try:
        m = experts.get("Marker", {}) or {}
        v = experts.get("VLM", {}) or {}
        s = experts.get("Spatial", {}) or {}
        if s.get("label_support"):
            m_ls = m.get("label_support", {}) or {}
            v_ls = v.get("label_support", {}) or {}
            surface_lbls = ["Layer 1", "Mixed L1/L2", "Layer 2"]
            m_surface = max(_safe_float(m_ls.get(lbl, 0.0)) for lbl in surface_lbls)
            v_surface = max(_safe_float(v_ls.get(lbl, 0.0)) for lbl in surface_lbls)
            s_top, s_sc = _top_label(s)
            if float(m_surface) >= 0.50 and float(v_surface) >= 0.60 and float(s_sc) >= 0.60:
                if s_top in {"Mixed L2/L3", "Layer 3", "Mixed L3/L4"}:
                    if "Spatial" not in rr:
                        rr.append("Spatial")
                    guidance_by_agent.setdefault(
                        "Spatial",
                        "Peer-consensus indicates this domain is superficial (Layer 1 / Mixed L1/L2) based on Visual+Marker. "
                        "Re-evaluate your depth mapping and thresholds: confirm normalized_depth calibration (depth_flipped) "
                        "and treat boundary cases conservatively. Prefer Mixed L1/L2 if depth is near superficial boundary. "
                        "Use ONLY spatial_summary fields in reasoning.",
                    )
                    reason_parts.append("conflict_surface_vs_spatial: rerun Spatiality")
    except Exception:
        pass

    return rr, guidance_by_agent, reason_parts


_MYELIN_MARKERS = {
    "MBP",
    "PLP1",
    "MOG",
    "MAG",
    "MOBP",
    "CNP",
    "CLDN11",
    "UGT8",
    "SOX10",
    "MYRF",
}


def _marker_myelin_hits(marker_out: ExpertOutput) -> List[str]:
    hits: List[str] = []
    for ev in marker_out.get("evidence", []) or []:
        if ev.get("type") != "deg_marker":
            continue
        for g in ev.get("items", []) or []:
            gu = str(g).upper()
            if gu in _MYELIN_MARKERS and gu not in hits:
                hits.append(gu)
    return hits


def _wm_supported_by_other_experts(experts: Dict[str, ExpertOutput], *, min_support: float = 0.25) -> bool:
    """Require at least one non-marker expert to also support White Matter."""
    for a in ["VLM", "Spatial"]:
        sup = (experts.get(a, {}) or {}).get("label_support", {}) or {}
        if _safe_float(sup.get("White Matter", 0.0)) >= float(min_support):
            return True
    return False


def _apply_white_matter_guardrail(
    annotation: Dict[str, Any],
    experts: Dict[str, ExpertOutput],
    *,
    label_space: List[str],
    weights: Dict[str, float],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Prevent false-positive 'White Matter' labels, but do NOT over-reject true WM.

    Design intent:
    - Visual/Spatiality are weak at detecting myelin; Marker is strong.
    - Therefore, we ALLOW 'White Matter' when marker evidence is strong enough,
      even if Visual/Spatiality do not explicitly vote for WM.

    Current rule:
    - If annotator label is White Matter, ALLOW it when:
      (A) >=2 myelin markers in Marker evidence AND Marker WM support >= 0.75
          OR
      (B) >=2 myelin markers AND Visual/Spatiality has non-trivial WM support (>=0.15)
          OR
      (C) Visual WM morphology support is strong (>=0.70) AND Marker WM support >= 0.75 AND oligodendrocyte support >= 0.85
    - Otherwise, revise label away from White Matter to the best non-WM label
      (prefer Visual/Spatiality top label).
    """
    if str(annotation.get("biological_identity", "")) != "White Matter":
        return annotation, None

    marker = experts.get("Marker", {}) or {}
    myelin_hits = _marker_myelin_hits(marker)
    marker_ls = marker.get("label_support", {}) or {}
    marker_ct = marker.get("celltype_support", {}) or {}
    marker_wm = _safe_float(marker_ls.get("White Matter", 0.0))
    oligo = _safe_float(marker_ct.get("Oligodendrocytes", 0.0))
    supported_elsewhere = _wm_supported_by_other_experts(experts, min_support=0.15)
    vlm_ls = (experts.get("VLM", {}) or {}).get("label_support", {}) or {}
    vlm_wm = _safe_float(vlm_ls.get("White Matter", 0.0))

    # Allow pure White Matter ONLY when evidence is strong enough to distinguish it from
    # WM-adjacent deep gray (common in DLPFC).
    #
    # - >=3 myelin markers: strong molecular support
    # - OR >=2 myelin markers AND Visual/Spatiality also supports WM (morphology/depth)
    # - OR Visual WM morphology is very strong + strong marker support
    allow = False
    if len(myelin_hits) >= 3 and marker_wm >= 0.75:
        allow = True
    if len(myelin_hits) >= 2 and supported_elsewhere:
        allow = True
    if vlm_wm >= 0.70 and marker_wm >= 0.75 and oligo >= 0.85:
        allow = True

    if allow:
        return annotation, None

    # If WM is rejected, prefer a WM-adjacent boundary label when marker evidence is strong but
    # Visual/Spatiality do NOT support true WM morphology. This reduces systematic over-calling of pure WM
    # for deep layer / WM-adjacent gray (common in DLPFC).
    if "Mixed L6/White Matter" in set(label_space) and len(myelin_hits) >= 2 and marker_wm >= 0.75 and not supported_elsewhere and vlm_wm < 0.70:
        alt_lbl = "Mixed L6/White Matter"
        alt_sc = 0.65
    else:
        # Otherwise, follow Visual/Spatiality top label (layer morphology / depth),
        # rather than letting global aggregation drift to unrelated superficial labels.
        alt_lbl = ""
        alt_sc = 0.0
        candidates: Dict[str, float] = {}
        for a in ["VLM", "Spatial"]:
            sup = (experts.get(a, {}) or {}).get("label_support", {}) or {}
            for lbl, v in sup.items():
                if str(lbl) == "White Matter":
                    continue
                try:
                    candidates[str(lbl)] = max(float(candidates.get(str(lbl), 0.0)), float(v))
                except Exception:
                    continue
        if candidates:
            alt_lbl = sorted(candidates.items(), key=lambda t: float(t[1]), reverse=True)[0][0]
            alt_sc = float(candidates.get(alt_lbl, 0.0))
        else:
            alt_lbl, alt_sc = _best_label_excluding(experts, label_space, weights, exclude=["White Matter"])
    revised = dict(annotation)
    revised["biological_identity"] = alt_lbl
    revised["biological_identity_conf"] = float(max(0.25, min(1.0, float(alt_sc))))
    # prepend reasoning note (keep length bounded)
    note = (
        f"WM_GUARD: rejected White Matter because myelin_markers={myelin_hits}, "
        f"marker_wm={marker_wm:.2f}, oligo={oligo:.2f}, wm_supported_by_visual_or_spatiality={supported_elsewhere}. "
        f"Revised to {alt_lbl}. "
    )
    try:
        revised["reasoning"] = (note + str(revised.get("reasoning", "")))[:1500]
    except Exception:
        pass
    return revised, note


def _strong_wm_evidence(experts: Dict[str, ExpertOutput]) -> bool:
    """Heuristic for true white matter: strong myelin/oligo + often Visual WM morphology."""
    marker = experts.get("Marker", {}) or {}
    myelin_hits = _marker_myelin_hits(marker)
    marker_ls = marker.get("label_support", {}) or {}
    marker_ct = marker.get("celltype_support", {}) or {}
    marker_wm = _safe_float(marker_ls.get("White Matter", 0.0))
    oligo = _safe_float(marker_ct.get("Oligodendrocytes", 0.0))
    vlm_ls = (experts.get("VLM", {}) or {}).get("label_support", {}) or {}
    vlm_wm = _safe_float(vlm_ls.get("White Matter", 0.0))

    # Prefer >=3 myelin/oligo genes for calling true WM. (MBP+PLP1 alone is often WM-adjacent gray.)
    if len(myelin_hits) >= 3 and marker_wm >= 0.75:
        return True
    if len(myelin_hits) >= 1 and oligo >= 0.85 and marker_wm >= 0.85:
        return True
    if vlm_wm >= 0.60 and marker_wm >= 0.70:
        return True
    return False


def _downgrade_false_wm_trap(critic: CriticOutput, *, strong_wm: bool) -> CriticOutput:
    """If WM evidence is strong, a 'white_matter_trap' blocker is likely a false positive."""
    if not strong_wm:
        return critic
    try:
        issues = list(critic.get("issues", []) or [])
    except Exception:
        return critic
    changed = False
    new_issues = []
    for it in issues:
        try:
            if str(it.get("type", "")).strip() == "white_matter_trap" and bool(it.get("blocker")):
                it = dict(it)
                it["blocker"] = False
                # keep severity but make it advisory
                it["fix_hint"] = (str(it.get("fix_hint", "")) + " (auto-downgraded: strong WM evidence)").strip()
                changed = True
        except Exception:
            pass
        new_issues.append(it)
    if not changed:
        return critic
    out = dict(critic)
    out["issues"] = new_issues
    return out  # type: ignore[return-value]


def _is_blocker_gate(gate: Dict[str, Any]) -> bool:
    try:
        return "blocker_issue" in list(gate.get("fail_reasons") or [])
    except Exception:
        return False


def _better_attempt(prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> Dict[str, Any]:
    """Pick better attempt: prefer passed > non-blocker > higher final_score > higher critic_score."""
    if prev is None:
        return cur
    p_gate = prev.get("gate", {}) or {}
    c_gate = cur.get("gate", {}) or {}
    prev_key = (
        1 if bool(p_gate.get("passed")) else 0,
        1 if not _is_blocker_gate(p_gate) else 0,
        float(_safe_float(p_gate.get("final_score", 0.0))),
        float(_safe_float((prev.get("critic") or {}).get("critic_score", 0.0))),
    )
    cur_key = (
        1 if bool(c_gate.get("passed")) else 0,
        1 if not _is_blocker_gate(c_gate) else 0,
        float(_safe_float(c_gate.get("final_score", 0.0))),
        float(_safe_float((cur.get("critic") or {}).get("critic_score", 0.0))),
    )
    return cur if cur_key > prev_key else prev


def _decide_next_actions(
    *,
    annotation: Dict[str, Any],
    experts: Dict[str, ExpertOutput],
    critic: CriticOutput,
    label_space: List[str],
    weights: Dict[str, float],
) -> Tuple[Optional[Dict[str, Any]], List[str], str, Dict[str, str]]:
    """Return (revised_annotation, rerun_agents, reason).

    - revised_annotation: revise-only proposal for next round (no expert rerun required)
    - rerun_agents: minimal experts to rerun next round
    """
    issues = list(critic.get("issues", []) or [])
    rr_agents: List[str] = []
    reason_parts: List[str] = []
    guidance_by_agent: Dict[str, str] = {}

    # 1) Respect explicit rerun_agent hints from critic issues
    for it in issues:
        ra = (it.get("rerun_agent") or "").strip()
        if ra in {"Marker", "Pathway", "Spatial", "VLM"} and ra not in rr_agents:
            rr_agents.append(ra)

    # 2) Prefer rerunning the outlier expert on label conflicts (instead of revise-only).
    rr_agents, guidance_by_agent, conflict_reasons = _prefer_rerun_over_revise_for_conflict(
        experts=experts,
        label_space=label_space,
        weights=weights,
        existing_reruns=rr_agents,
        guidance_by_agent=guidance_by_agent,
        issues=issues,
    )
    for cr in conflict_reasons:
        if cr:
            reason_parts.append(cr)

    # 3) Try revise-only when critic hints a target label
    # IMPORTANT: only use explicit fix hints for auto label revision.
    # Do NOT parse critic.reasoning, because it frequently mentions labels in NEGATION
    # (e.g., "White Matter is invalid"), which can cause incorrect revise-only actions.
    hint_text = " ".join([str(it.get("fix_hint", "")) for it in issues if it.get("fix_hint")])
    suggested = _extract_suggested_labels(hint_text, label_space)
    revised: Optional[Dict[str, Any]] = None
    # If we plan to rerun experts to resolve conflict, do not revise label in the same step.
    if rr_agents:
        suggested = []
    if suggested:
        target = suggested[0]
        if target and target != annotation.get("biological_identity"):
            revised = dict(annotation)
            revised["biological_identity"] = target
            # keep conf conservative: set to max(current_conf, best_support_for_target) capped
            try:
                # approximate support for target from current experts
                best_lbl, best_sc = _best_label_excluding(experts, label_space, weights, exclude=[])
                # compute a pseudo support for target by temporarily forcing it as best if it exists
                # If target isn't present in any support, keep existing conf.
                # (We avoid heavy recompute here; annotator will still log its own conf.)
                if target == best_lbl:
                    revised["biological_identity_conf"] = float(max(float(annotation.get("biological_identity_conf", 0.0)), best_sc))
                else:
                    revised["biological_identity_conf"] = float(max(0.25, float(annotation.get("biological_identity_conf", 0.0)) * 0.8))
            except Exception:
                pass
            reason_parts.append(f"revise_only: set label -> {target}")

    # 4) White Matter trap handling:
    # - If WM evidence is weak and annotator chose WM, revise away from WM and rerun.
    # - If WM evidence is strong, keep WM (do not revise away); rerun Spatiality if needed to reconcile priors.
    for it in issues:
        if str(it.get("type", "")).strip() == "white_matter_trap":
            if str(annotation.get("biological_identity", "")) == "White Matter":
                marker = experts.get("Marker", {}) or {}
                myelin_hits = _marker_myelin_hits(marker)
                marker_ls = marker.get("label_support", {}) or {}
                marker_ct = marker.get("celltype_support", {}) or {}
                marker_wm = _safe_float(marker_ls.get("White Matter", 0.0))
                oligo = _safe_float(marker_ct.get("Oligodendrocytes", 0.0))
                vlm_ls = (experts.get("VLM", {}) or {}).get("label_support", {}) or {}
                vlm_wm = _safe_float(vlm_ls.get("White Matter", 0.0))

                strong_wm = False
                if len(myelin_hits) >= 2 and marker_wm >= 0.75:
                    strong_wm = True
                if len(myelin_hits) >= 1 and oligo >= 0.85 and marker_wm >= 0.85:
                    strong_wm = True
                if vlm_wm >= 0.60 and marker_wm >= 0.70:
                    strong_wm = True

                if strong_wm:
                    # keep WM; reconcile with spatial by rerunning Spatiality only
                    if "Spatial" not in rr_agents:
                        rr_agents.append("Spatial")
                    reason_parts.append("white_matter_trap: keep WM (strong marker/Visual evidence); rerun Spatiality")
                    break

                # weak WM -> revise away
                alt_lbl, _ = _best_label_excluding(experts, label_space, weights, exclude=["White Matter"])
                revised = dict(annotation)
                revised["biological_identity"] = alt_lbl
                revised["biological_identity_conf"] = float(
                    max(0.25, float(annotation.get("biological_identity_conf", 0.0)) * 0.7)
                )
                reason_parts.append(f"white_matter_trap: revise label -> {alt_lbl}")

            for a in ["Marker", "Spatial", "VLM"]:
                if a not in rr_agents:
                    rr_agents.append(a)
            reason_parts.append("white_matter_trap: rerun Marker+Spatiality+Visual")
            break

    # 5) Label conflict/low confidence: handled by conflict policy above; keep for bookkeeping only
    for it in issues:
        t = str(it.get("type", "")).strip()
        if t in {"label_conflict", "label_confidence_low"}:
            if revised is None:
                # choose best Mixed label if available
                mixed_pref = [l for l in label_space if l.startswith("Mixed ")]
                for cand in mixed_pref:
                    if cand in suggested:
                        continue
                # fallback: keep as-is
            reason_parts.append(t)

    # 6) If no clear action, rerun weakest-quality experts (targeted)
    if not rr_agents and revised is None:
        for a in ["VLM", "Spatial", "Marker", "Pathway"]:
            out = experts.get(a, {}) or {}
            flags = set(out.get("quality_flags", []) or [])
            score = _safe_float(out.get("agent_score", 0.0))
            if {"missing", "hallucinated_marker_evidence", "hallucinated_pathway_evidence"} & flags:
                rr_agents.append(a)
            elif score < 0.50:
                rr_agents.append(a)
        if rr_agents:
            reason_parts.append("heuristic_rerun_low_quality")

    # 7) Expert disagreement reruns (generic; works for any agent with label_support)
    # Keep as a fallback for non-conflict cases.
    if not rr_agents and revised is None:
        dis_reruns, dis_guidance, dis_reasons = _detect_disagreement_reruns(
            experts, label_space=label_space, weights=weights
        )
        for a in dis_reruns:
            if a not in rr_agents:
                rr_agents.append(a)
        guidance_by_agent.update(dis_guidance)
        reason_parts.extend(dis_reasons)

    reason = "; ".join([p for p in reason_parts if p])[:400]
    return revised, rr_agents, reason, guidance_by_agent


def _postcheck_marker_output(out: ExpertOutput, allowed_genes: List[str]) -> ExpertOutput:
    """Lightweight guardrail: ensure marker evidence uses only provided genes."""
    gene_set = {g.upper() for g in allowed_genes if isinstance(g, str)}
    bad = False
    for ev in out.get("evidence", []) or []:
        if ev.get("type") != "deg_marker":
            continue
        items = ev.get("items", []) or []
        for g in items:
            if str(g).upper() not in gene_set:
                bad = True
                break
        if bad:
            break
    if bad:
        # down-weight and flag
        out = dict(out)
        out["agent_score"] = float(_safe_float(out.get("agent_score", 0.0)) * 0.2)
        flags = list(out.get("quality_flags", []) or [])
        flags.append("hallucinated_marker_evidence")
        out["quality_flags"] = flags

    # If marker evidence contains only generic neuronal/cytoskeletal genes and no layer-specific markers,
    # treat it as "non-specific for layer" so it doesn't dominate peer confidence.
    try:
        layer_markers = {
            "RELN",
            "CXCL14",
            "CPLX3",
            "CUX1",
            "CUX2",
            "CALB1",
            "SATB2",
            "RORB",
            "PCP4",
            "BCL11B",
            "FEZF2",
            "TSHZ2",
            "TLE4",
            "FOXP2",
            "CRYM",
            "MBP",
            "PLP1",
            "MOG",
            "MAG",
            "MOBP",
            "CNP",
            "CLDN11",
        }
        nonspecific = {"NEFL", "NEFM", "MAP1B", "TUBA1B", "TUBA1A", "TUBB", "TUBB3", "ENO2"}
        mentioned: List[str] = []
        for ev in out.get("evidence", []) or []:
            if ev.get("type") != "deg_marker":
                continue
            for g in ev.get("items", []) or []:
                mentioned.append(str(g).upper())
        mentioned_set = {g for g in mentioned if g}
        has_layer = any(g in layer_markers for g in mentioned_set)
        has_specific = any(g in layer_markers for g in mentioned_set if g not in {"MBP", "PLP1", "MOG", "MAG", "MOBP", "CNP", "CLDN11"})
        mostly_nonspecific = len(mentioned_set) > 0 and (len([g for g in mentioned_set if g in nonspecific]) / max(1, len(mentioned_set))) >= 0.5
        if (not has_specific) and mostly_nonspecific:
            out = dict(out)
            flags = list(out.get("quality_flags", []) or [])
            if "non_specific_layer_markers" not in flags:
                flags.append("non_specific_layer_markers")
            out["quality_flags"] = flags
    except Exception:
        pass
    return out


def _postcheck_pathway_output(out: ExpertOutput, allowed_terms: List[str]) -> ExpertOutput:
    term_set = {t.lower() for t in allowed_terms if isinstance(t, str)}
    bad = False
    for ev in out.get("evidence", []) or []:
        if ev.get("type") != "pathway":
            continue
        items = ev.get("items", []) or []
        for t in items:
            # allow partial matches
            if str(t).lower() not in term_set:
                bad = True
                break
        if bad:
            break
    if bad:
        out = dict(out)
        out["agent_score"] = float(_safe_float(out.get("agent_score", 0.0)) * 0.3)
        flags = list(out.get("quality_flags", []) or [])
        flags.append("hallucinated_pathway_evidence")
        out["quality_flags"] = flags
    return out


def _pathway_priors(
    top_pathways: List[Dict[str, Any]], label_space: List[str]
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """Compute a deterministic prior from pathway keywords.

    Note: Pathway is NOT used for layer prediction anymore, but we keep label priors
    for potential future use/debugging; current pipeline only uses ct_support + evidence.
    """
    text = " | ".join([str(p.get("term", "")) for p in top_pathways]).lower()
    # keyword buckets
    myelin = any(k in text for k in ["myelin", "oligodend", "white matter", "ensheath"])
    synapse = any(k in text for k in ["synap", "potentiation", "calcium signaling", "glutamat", "gaba", "axon guidance"])
    neurodeg = any(k in text for k in ["neurodegeneration", "alzheimer", "parkinson", "huntington", "oxidative phosphorylation"])
    immune = any(k in text for k in ["immune", "inflamm", "complement", "cytokine"])

    # base scores
    s: Dict[str, float] = {lbl: 0.0 for lbl in label_space}
    ct: Dict[str, float] = {}
    evidence: List[Dict[str, Any]] = []

    if myelin:
        s["White Matter"] = max(s.get("White Matter", 0.0), 0.55)
        s["Layer 6"] = max(s.get("Layer 6", 0.0), 0.25)
        s["Mixed L5/L6"] = max(s.get("Mixed L5/L6", 0.0), 0.20)
        ct["Oligodendrocytes"] = 0.75
        evidence.append({"id": "PP1", "type": "pathway_prior", "items": ["myelin/oligodendrocyte keywords"], "direction": "prior", "strength": 0.9})

    if synapse or neurodeg:
        # neuronal activity is common in mid/deep layers; keep it broad
        s["Layer 3"] = max(s.get("Layer 3", 0.0), 0.35)
        s["Layer 5"] = max(s.get("Layer 5", 0.0), 0.25)
        s["Layer 4"] = max(s.get("Layer 4", 0.0), 0.15)
        s["Mixed L2/L3"] = max(s.get("Mixed L2/L3", 0.0), 0.15)
        ct["Excitatory neurons"] = max(ct.get("Excitatory neurons", 0.0), 0.70)
        evidence.append({"id": "PP2", "type": "pathway_prior", "items": ["synaptic/neuron-activity keywords"], "direction": "prior", "strength": 0.8})

    if immune:
        # immune signals are not layer-specific; increase Mixed/Unknown
        s["Mixed"] = max(s.get("Mixed", 0.0), 0.40)
        s["Unknown"] = max(s.get("Unknown", 0.0), 0.20)
        ct["Microglia/macrophages"] = max(ct.get("Microglia/macrophages", 0.0), 0.55)
        evidence.append({"id": "PP3", "type": "pathway_prior", "items": ["immune/inflammation keywords"], "direction": "prior", "strength": 0.7})

    # Normalize to top-4 labels only
    ranked = sorted(s.items(), key=lambda t: float(t[1]), reverse=True)
    ranked = [(k, v) for k, v in ranked if v > 0][:4]
    if not ranked:
        ranked = [("Unknown", 1.0)]
    total = sum(v for _, v in ranked) or 1.0
    lbl_support = {k: float(v / total) for k, v in ranked}

    # celltypes: keep top-3
    ranked_ct = sorted(ct.items(), key=lambda t: float(t[1]), reverse=True)[:3]
    total_ct = sum(v for _, v in ranked_ct) or 1.0
    ct_support = {k: float(v / total_ct) for k, v in ranked_ct} if ranked_ct else {}

    return lbl_support, ct_support, evidence


def _summarize_biological_function(
    client: Any,
    model: str,
    *,
    domain_id: int,
    annotation: Dict[str, Any],
    pathways_df: pd.DataFrame,
    language: str = "en",
) -> str:
    """Ask LLM for a short biological function summary for this domain."""
    try:
        top_pathways = _get_top_pathways(pathways_df, domain_id, top_n=5)
        context = {
            "domain_id": int(domain_id),
            "biological_identity": annotation.get("biological_identity"),
            "primary_cell_types": annotation.get("primary_cell_types", []),
            "key_evidence": annotation.get("key_evidence", []),
            "top_pathways": top_pathways,
        }
        lang = str(language or "en").lower()
        if lang.startswith("zh"):
            instr = (
                "用简洁的中文总结该空间域的大致生物学功能，1-2 句，不超过 60 个汉字。"
            )
        else:
            instr = (
                "Summarize the main biological function of this spatial domain in 1–2 very short English sentences (<= 40 words)."
            )

        prompt = (
            "You are an expert in spatial transcriptomics.\n"
            "Based on the following annotation and pathway context, write a very short description of the biological function of this domain.\n"
            f"{instr}\n\n"
            "Input JSON:\n"
            f"{json.dumps(context, ensure_ascii=False)}\n\n"
            "Return ONLY JSON of the form:\n"
            '{"function": "<short description>"}'
        )

        resp = chat_json(
            client,
            model,
            [
                {"role": "system", "content": "Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.2,
        )
        fn = str(resp.get("function", "")).strip()
        return fn[:300]
    except Exception:
        return ""


def run_annotation_multiagent(
    *,
    data_dir: str,
    sample_id: str,
    target_domains: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    kb_path: Optional[str] = None,
    config: Optional[ScoreRagConfig] = None,
) -> List[Dict[str, Any]]:
    """Run multi-agent annotation for one sample and save per-round JSON logs.

    Returns the final list of domain annotations (also written to output_dir/domain_annotations.json).
    """
    cfg = config or load_config()
    settings = AnnotationSettings(
        label_space=list(getattr(cfg, "annotation_label_space", None) or DEFAULT_LABEL_SPACE),
        weights=cfg.annotation_weights,
        standard_score=float(cfg.annotation_standard_score),
        max_rounds=int(cfg.annotation_max_rounds),
        language=cfg.annotation_language,
        log_dir=cfg.annotation_log_dir,
        vlm_required=bool(cfg.annotation_vlm_required),
        vlm_min_score=float(cfg.annotation_vlm_min_score),
        vlm_image_path=cfg.annotation_vlm_image_path,
    )

    if output_dir is None:
        # default: sibling annotation_output next to data_dir
        output_dir = os.path.join(os.path.dirname(data_dir), "annotation_output")
    os.makedirs(output_dir, exist_ok=True)

    if kb_path is None:
        kb_path = str(getattr(cfg, "annotation_kb_path", "") or "")

    degs_df, pathways_df, spots_df = _load_inputs(data_dir, sample_id)
    all_domains = sorted(set(int(x) for x in spots_df["spatial_domain"].unique()))
    domains = [d for d in all_domains if (not target_domains or d in target_domains)]

    # Provider client (text+vision depending on selected provider/model capability)
    client = make_provider_client(
        api_provider=getattr(cfg, "api_provider", ""),
        api_key=getattr(cfg, "api_key", ""),
        api_endpoint=getattr(cfg, "api_endpoint", ""),
        api_version=getattr(cfg, "api_version", ""),
        api_model=getattr(cfg, "api_model", ""),
        azure_openai_key=getattr(cfg, "azure_openai_key", ""),
        azure_endpoint=getattr(cfg, "azure_endpoint", ""),
        azure_api_version=getattr(cfg, "azure_api_version", ""),
        azure_deployment=getattr(cfg, "azure_deployment", ""),
    )
    model = str(getattr(cfg, "api_model", "") or getattr(cfg, "azure_deployment", ""))

    # Run folder for this sample
    rd = run_dir(settings.log_dir, sample_id)

    # --- Spatiality calibration (Visual-anchored depth orientation) ---
    # Compute raw depths with PCA if possible, else y-normalized. Then use Visual to identify the
    # most "surface-like" domain (Layer 1/2) among the requested domains; if that domain appears
    # deep in the raw depth axis, flip the depth axis so that 0=superficial and 1=deep.
    depth_model = _compute_depth_model_pca(spots_df)
    domain_depths, depth_method = _compute_domain_depths(spots_df, all_domains, depth_model)
    domain_depths_raw = dict(domain_depths)

    vlm_path = _resolve_vlm_image_path(settings.vlm_image_path, data_dir=data_dir, sample_id=sample_id)
    vlm_cache_by_domain: Dict[int, ExpertOutput] = {}
    surface_domain: Optional[int] = None
    surface_score_best = -1.0
    # IMPORTANT: when running a subset of domains (e.g., [5,6]), using only `domains` to find a
    # superficial anchor can fail (no L1/2 present -> surface_score=0). For small label spaces
    # (like 151507 with 7 domains), use all_domains for calibration to keep depth axis stable.
    anchor_domains = all_domains if len(all_domains) <= 20 else domains
    for d in anchor_domains:
        v = vlm_agent(client, model, domain_id=int(d), image_path=vlm_path, language=settings.language)
        vlm_cache_by_domain[int(d)] = v
        if not _vlm_failed(v, settings.vlm_min_score):
            s = _surface_score_from_vlm(v)
            if s > surface_score_best:
                surface_score_best = s
                surface_domain = int(d)

    depth_flipped = False
    # Only flip when we have a confident surface anchor; otherwise (e.g., single-domain runs)
    # the "surface" detection may be too weak and flipping would be harmful.
    if surface_score_best >= 0.60 and surface_domain is not None and surface_domain in domain_depths:
        try:
            surf_depth = float(domain_depths[surface_domain])
            if surf_depth > 0.5:
                depth_flipped = True
                for k in list(domain_depths.keys()):
                    domain_depths[k] = float(1.0 - float(domain_depths[k]))
        except Exception:
            depth_flipped = False

    # --- Optional: adjacency-based smoothing for layer-order consistency (experimental) ---
    # This is intentionally feature-flagged; if it doesn't help, keep it disabled.
    adj_enabled = bool(getattr(cfg, "annotation_spatial_adjacency_smoothing", False))
    adj_knn_k = int(getattr(cfg, "annotation_spatial_knn_k", 8))
    adj_alpha = float(getattr(cfg, "annotation_spatial_smoothing_alpha", 0.35))
    adj_iters = int(getattr(cfg, "annotation_spatial_smoothing_iters", 3))
    adj_min_edges = int(getattr(cfg, "annotation_spatial_adjacency_min_edges", 30))

    domain_adjacency: Dict[int, Dict[int, float]] = {}
    domain_depths_smoothed: Optional[Dict[int, float]] = None
    if adj_enabled:
        domain_adjacency = _build_domain_adjacency_knn(spots_df, k=adj_knn_k, min_edges=adj_min_edges)
        if domain_adjacency:
            domain_depths_smoothed = _smooth_depths_by_adjacency(domain_depths, domain_adjacency, alpha=adj_alpha, iters=adj_iters)
            # Use smoothed depths for spatial summary only (do not alter raw dict for other logic)
        else:
            domain_depths_smoothed = None

    final_results: List[Dict[str, Any]] = []

    for domain_id in domains:
        # cache per expert (reused across rounds unless rerun requested)
        expert_cache: Dict[str, ExpertOutput] = {}
        # Seed Visual so round-1 doesn't re-call Visual unless it failed or critic requests rerun.
        if int(domain_id) in vlm_cache_by_domain:
            expert_cache["VLM"] = vlm_cache_by_domain[int(domain_id)]
        last_best: Dict[str, Any] = {}
        best_attempt: Optional[Dict[str, Any]] = None
        pending_revision: Optional[Dict[str, Any]] = None
        planned_rerun_agents: Optional[set] = None
        planned_reason: str = ""
        planned_guidance: Optional[Dict[str, str]] = None

        for round_idx in range(1, settings.max_rounds + 1):
            # Determine which experts to run this round
            rerun_agents = set()
            if round_idx == 1:
                rerun_agents = {"Marker", "Pathway", "Spatial", "VLM"}
            else:
                # carry over planned reruns from previous round (may come from critic or auto-policy)
                rerun_agents = set(planned_rerun_agents or [])
                planned_rerun_agents = None

            # Always rerun Visual if required and last Visual failed
            if settings.vlm_required:
                prev_vlm = expert_cache.get("VLM")
                if prev_vlm is None or _vlm_failed(prev_vlm, settings.vlm_min_score):
                    rerun_agents.add("VLM")
                else:
                    # If we already have a valid Visual output, don't rerun in round 1 by default.
                    if round_idx == 1 and "VLM" in rerun_agents:
                        rerun_agents.discard("VLM")

            # Run required experts (or reuse cache)
            if "Marker" in rerun_agents:
                # Use a wider top-N so layer-specific markers (often not highest logFC) can be included.
                top_degs = _get_top_degs(degs_df, domain_id, top_n=30)
                expert_cache["Marker"] = marker_celltype_agent(
                    client,
                    model,
                    domain_id=domain_id,
                    top_degs=top_degs,
                    rerun_guidance=(planned_guidance or {}).get("Marker"),
                    language=settings.language,
                )
                expert_cache["Marker"] = _postcheck_marker_output(expert_cache["Marker"], [d["gene"] for d in top_degs])

            if "Pathway" in rerun_agents:
                top_pws = _get_top_pathways(pathways_df, domain_id, top_n=5)
                pri_lbl, pri_ct, pri_ev = _pathway_priors(top_pws, settings.label_space)
                expert_cache["Pathway"] = pathway_agent(
                    client,
                    model,
                    domain_id=domain_id,
                    top_pathways=top_pws,
                    prior_celltype_support=pri_ct,
                    rerun_guidance=(planned_guidance or {}).get("Pathway"),
                    language=settings.language,
                )
                # inject deterministic prior evidence so critic can cite it
                try:
                    ev = list(expert_cache["Pathway"].get("evidence", []) or [])
                    expert_cache["Pathway"]["evidence"] = (pri_ev + ev)[:6]
                except Exception:
                    pass
                expert_cache["Pathway"] = _postcheck_pathway_output(expert_cache["Pathway"], [p["term"] for p in top_pws])

            if "Spatial" in rerun_agents:
                # Build a compact adjacency diagnostic for this domain (top neighbors only)
                adj_list: List[Dict[str, Any]] = []
                try:
                    nbrs = (domain_adjacency.get(int(domain_id), {}) or {}) if adj_enabled else {}
                    for n, w in sorted(nbrs.items(), key=lambda t: float(t[1]), reverse=True)[:5]:
                        adj_list.append({"domain": int(n), "weight": round(float(w), 4)})
                except Exception:
                    adj_list = []

                d_raw = domain_depths.get(int(domain_id))
                d_use = (domain_depths_smoothed or {}).get(int(domain_id)) if domain_depths_smoothed else d_raw
                spatial_summary = _compute_spatial_summary(
                    spots_df,
                    domain_id,
                    domain_depth=d_use,
                    domain_depth_raw=d_raw,
                    adjacent_domains=adj_list,
                    depth_method=(depth_method + ("_flipped" if depth_flipped else "")),
                )
                expert_cache["Spatial"] = spatial_anatomy_agent(
                    client,
                    model,
                    domain_id=domain_id,
                    spatial_summary=spatial_summary,
                    rerun_guidance=(planned_guidance or {}).get("Spatial"),
                    language=settings.language,
                )

            if "VLM" in rerun_agents:
                expert_cache["VLM"] = vlm_agent(
                    client,
                    model,
                    domain_id=domain_id,
                    image_path=vlm_path,
                    rerun_guidance=(planned_guidance or {}).get("VLM"),
                    language=settings.language,
                )

            # Ensure all experts exist (fallback stubs)
            for a in ["Marker", "Pathway", "Spatial", "VLM"]:
                expert_cache.setdefault(
                    a,
                    {
                        "agent": a,  # type: ignore[typeddict-item]
                        "agent_score": 0.0,
                        # Pathway does not participate in layer prediction.
                        "label_support": ({} if a == "Pathway" else {"Unknown": 1.0}),
                        "celltype_support": {},
                        "reasoning": f"{a} not executed",
                        "evidence": [{"id": f"{a}0", "type": "missing", "items": [a], "strength": 1.0}],
                        "quality_flags": ["missing"],
                    },
                )

            # Propose (or use pending revise-only draft)
            if pending_revision is not None:
                annotation = pending_revision
                pending_revision = None
            else:
                annotation = propose_annotation(
                    domain_id,
                    expert_cache,
                    weights=settings.weights,
                    label_space=settings.label_space,
                    topk_alternatives=2,
                    topk_celltypes=6,
                )

            # Hard guardrail: avoid 'White Matter' unless supported by multiple myelin markers AND Visual/Spatiality.
            wm_guard_note = None
            if bool(getattr(cfg, "annotation_wm_guard_enabled", True)):
                annotation, wm_guard_note = _apply_white_matter_guardrail(
                    annotation, expert_cache, label_space=settings.label_space, weights=settings.weights
                )

            # Critic
            critic = critic_agent(
                client,
                model,
                annotation=annotation,
                experts=expert_cache,
                kb_path=kb_path,
                standard_score=settings.standard_score,
                language=settings.language,
            )
            # Deterministic safety: if the proposed label is White Matter and evidence is strong,
            # downgrade any accidental 'white_matter_trap' blocker from the LLM critic.
            if str(annotation.get("biological_identity", "")) == "White Matter":
                critic = _downgrade_false_wm_trap(critic, strong_wm=_strong_wm_evidence(expert_cache))

            # Peer + gate score
            peer_weighted, peer_avg = compute_peer_scores(expert_cache, settings.weights)
            critic_score = _safe_float(critic.get("critic_score", 0.0))
            final_score = 0.8 * critic_score + 0.2 * float(peer_weighted)

            fail_reasons: List[str] = []
            passed = True

            # Visual required gate
            if settings.vlm_required and _vlm_failed(expert_cache.get("VLM", {}), settings.vlm_min_score):
                passed = False
                fail_reasons.append("vlm_required_failed")
                # force rerun request to Visual
                critic.setdefault("rerun_request", {})
                critic["rerun_request"]["rerun"] = True
                critic["rerun_request"]["agents"] = ["VLM"]
                critic["rerun_request"]["reason"] = "Visual required gate failed"

            if _has_blocker(critic, proposed_label=str(annotation.get("biological_identity", ""))):
                passed = False
                fail_reasons.append("blocker_issue")

            if final_score < settings.standard_score:
                passed = False
                fail_reasons.append("final_below_standard")

            round_log: RoundLog = {
                "run_info": {
                    "sample_id": sample_id,
                    "domain_id": int(domain_id),
                    "round": int(round_idx),
                    "timestamp": utc_now_iso(),
                    "label_space": settings.label_space,
                    "weights": settings.weights,
                    "standard_score": settings.standard_score,
                    "max_rounds": settings.max_rounds,
                    "vlm_required": settings.vlm_required,
                    "vlm_min_score": settings.vlm_min_score,
                    "vlm_image_path": settings.vlm_image_path,
                    "spatial_depth_method": depth_method,
                    "spatial_depth_flipped": depth_flipped,
                    "spatial_surface_domain": surface_domain,
                    "spatial_surface_score": round(float(surface_score_best), 3),
                    "spatial_adj_smoothing_enabled": adj_enabled,
                    "spatial_adj_knn_k": adj_knn_k,
                    "spatial_adj_min_edges": adj_min_edges,
                    "spatial_adj_alpha": round(float(adj_alpha), 4),
                    "spatial_adj_iters": int(adj_iters),
                    "loop_planned_reason": planned_reason,
                    "loop_planned_guidance": (planned_guidance or {}),
                    "wm_guardrail_applied": bool(wm_guard_note),
                },
                "experts": expert_cache,
                "proposer": {
                    "annotation": annotation,
                    "peer": {
                        "peer_score_weighted": float(peer_weighted),
                        "peer_score_unweighted_avg": float(peer_avg),
                    },
                },
                "critic": critic,
                "gate": {
                    "peer_score_weighted": float(peer_weighted),
                    "peer_score_unweighted_avg": float(peer_avg),
                    "final_score": float(final_score),
                    "passed": bool(passed),
                    "fail_reasons": fail_reasons,
                },
            }

            log_path = os.path.join(rd, f"domain_{domain_id}_round_{round_idx}.json")
            write_json(log_path, round_log)

            # record last attempt (for rerun decisions) + best attempt (for final output)
            last_best = {"annotation": annotation, "critic": critic, "gate": round_log["gate"], "log_path": log_path}
            best_attempt = _better_attempt(best_attempt, last_best)

            if passed:
                break

            # Decide next actions (fix missing loop closure).
            planned_reason = ""
            if round_idx < settings.max_rounds:
                rr = (critic.get("rerun_request") or {}) if isinstance(critic, dict) else {}
                rr_agents = list(rr.get("agents") or [])
                rr_agents = [a for a in rr_agents if a in {"Marker", "Pathway", "Spatial", "VLM"}]

                revised, auto_reruns, auto_reason, auto_guidance = _decide_next_actions(
                    annotation=annotation,
                    experts=expert_cache,
                    critic=critic,
                    label_space=settings.label_space,
                    weights=settings.weights,
                )

                # prefer critic-requested reruns when present; otherwise use auto_reruns
                next_reruns = rr_agents if rr_agents else auto_reruns
                pending_revision = revised
                planned_rerun_agents = set(next_reruns) if next_reruns else set()
                planned_reason = auto_reason or ("critic_rerun_request" if rr_agents else "")
                planned_guidance = auto_guidance or {}

                # Stop only if we truly have nothing actionable to do.
                if not planned_rerun_agents and pending_revision is None:
                    break
            else:
                break

        # finalize domain result with meta
        pick = best_attempt or last_best
        final_ann = dict(pick.get("annotation") or {})
        # Optional: LLM-generated short biological function summary for this domain.
        func_text = _summarize_biological_function(
            client,
            model,
            domain_id=int(domain_id),
            annotation=final_ann,
            pathways_df=pathways_df,
            language=settings.language,
        )
        if func_text:
            final_ann["function"] = func_text
        final_ann["meta"] = {
            "peer_score_weighted": pick.get("gate", {}).get("peer_score_weighted", None),
            "peer_score_unweighted_avg": pick.get("gate", {}).get("peer_score_unweighted_avg", None),
            "critic_score": (pick.get("critic") or {}).get("critic_score", None),
            "final_score": pick.get("gate", {}).get("final_score", None),
            "passed": pick.get("gate", {}).get("passed", None),
            "fail_reasons": pick.get("gate", {}).get("fail_reasons", []),
            "log_path": pick.get("log_path"),
            "selected_round": "best",
        }
        final_results.append(final_ann)

    # Write final outputs
    final_results.sort(key=lambda x: int(x.get("domain_id", 0)))
    out_path = os.path.join(output_dir, "domain_annotations.json")
    # If running a subset of domains, merge into existing file instead of overwriting everything.
    merged: List[Dict[str, Any]] = final_results
    if target_domains:
        results_map: Dict[int, Dict[str, Any]] = {}
        if os.path.exists(out_path):
            try:
                import json

                with open(out_path, "r", encoding="utf-8") as f:
                    existing = json.load(f) or []
                for item in existing:
                    if isinstance(item, dict) and "domain_id" in item:
                        try:
                            results_map[int(item["domain_id"])] = item
                        except Exception:
                            continue
            except Exception:
                results_map = {}
        for item in final_results:
            try:
                results_map[int(item.get("domain_id", 0))] = item
            except Exception:
                continue
        merged = sorted(results_map.values(), key=lambda x: int(x.get("domain_id", 0)))

    with open(out_path, "w", encoding="utf-8") as f:
        import json

        json.dump(merged, f, indent=2, ensure_ascii=False)

    return final_results

