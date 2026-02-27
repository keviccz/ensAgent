from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .schemas import AnnotationDraft, ExpertOutput


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(v) for v in weights.values())
    if s <= 0:
        return weights
    return {k: float(v) / s for k, v in weights.items()}


def _topk_items(d: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    return sorted(d.items(), key=lambda t: float(t[1]), reverse=True)[:k]


def _summarize_evidence(agent: str, item: Dict[str, Any]) -> str:
    """Build a short human-readable evidence string instead of opaque ids like 'M1'.

    Examples:
    - 'Marker: GAD1 (up)'
    - 'Pathway: Synaptic vesicle cycle (enriched)'
    - 'Spatial: normalized_radial_depth,density (prior)'
    - 'VLM: contiguous band,outer rim location (visual)'
    """
    try:
        items = [str(x) for x in (item.get("items") or []) if str(x)]
        items_str = ", ".join(items[:3])
        direction = str(item.get("direction") or "").strip().lower()
        dir_str = f" ({direction})" if direction else ""
        label = f"{agent}: "
        if items_str:
            label += items_str
        else:
            etype = str(item.get("type") or "").strip()
            note = str(item.get("note") or "").strip()
            if etype:
                label += etype
            if not items_str and not etype and note:
                label += note[:60]
        return (label + dir_str)[:120]
    except Exception:
        return str(agent)

def compute_peer_scores(experts: Dict[str, ExpertOutput], weights: Dict[str, float]) -> Tuple[float, float]:
    # Peer score is a confidence component, so we should not over-penalize when an expert
    # explicitly reports "non-specific" evidence for the target question.
    #
    # If an expert flags itself as non-specific (e.g., Marker has only generic neuron genes and
    # no layer-specific markers), we drop it from the peer-weight calculation and renormalize.
    eff_weights: Dict[str, float] = {}
    for agent, w in (weights or {}).items():
        out = experts.get(agent, {}) or {}
        flags = set(out.get("quality_flags", []) or [])
        if "non_specific_layer_markers" in flags:
            continue
        try:
            eff_weights[str(agent)] = float(w)
        except Exception:
            continue

    ws = _normalize_weights(eff_weights) if eff_weights else _normalize_weights(weights)

    scores = []
    weighted = 0.0
    for agent, out in experts.items():
        s = float(out.get("agent_score", 0.0))
        scores.append(s)
        weighted += ws.get(agent, 0.0) * s
    avg = sum(scores) / len(scores) if scores else 0.0
    return weighted, avg


def propose_annotation(
    domain_id: int,
    experts: Dict[str, ExpertOutput],
    *,
    weights: Dict[str, float],
    label_space: List[str],
    topk_alternatives: int = 2,
    topk_celltypes: int = 5,
) -> AnnotationDraft:
    # --- label aggregation ---
    # Pathway is not used for layer prediction; normalize weights only over agents that provide label_support.
    label_weights: Dict[str, float] = {}
    for agent, out in experts.items():
        sup = out.get("label_support", {}) or {}
        if sup:
            label_weights[agent] = float(weights.get(agent, 0.0))
    ws_label = _normalize_weights(label_weights)

    score_by_label: Dict[str, float] = {lbl: 0.0 for lbl in label_space}
    for agent, out in experts.items():
        sup = out.get("label_support", {}) or {}
        w = ws_label.get(agent, 0.0)
        for lbl, v in sup.items():
            if lbl not in score_by_label:
                continue
            try:
                score_by_label[lbl] += w * float(v)
            except Exception:
                continue

    ranked_labels = _topk_items(score_by_label, k=max(3, topk_alternatives + 1))
    best_label, best_conf = ranked_labels[0] if ranked_labels else ("Unknown", 0.0)
    second_conf = ranked_labels[1][1] if len(ranked_labels) > 1 else 0.0
    margin = float(best_conf - second_conf)

    alternatives = [{"label": lbl, "conf": float(sc)} for lbl, sc in ranked_labels[1 : 1 + topk_alternatives]]

    # --- cell type aggregation ---
    ws = _normalize_weights(weights)
    score_by_cell: Dict[str, float] = {}
    for agent, out in experts.items():
        sup = out.get("celltype_support", {}) or {}
        w = ws.get(agent, 0.0)
        for ct, v in sup.items():
            try:
                score_by_cell[ct] = score_by_cell.get(ct, 0.0) + w * float(v)
            except Exception:
                continue
    ranked_cells = _topk_items(score_by_cell, k=topk_celltypes)
    primary_cell_types = [f"{ct} (confidence={float(sc):.2f})" for ct, sc in ranked_cells]

    # --- evidence selection ---
    # Use short, human-readable evidence snippets instead of opaque ids like "M1".
    key_evidence: List[str] = []
    for agent in ["Marker", "Pathway", "Spatial", "VLM"]:
        items = experts.get(agent, {}).get("evidence", []) or []
        if not items:
            continue
        summary = _summarize_evidence(agent, items[0])
        if summary:
            key_evidence.append(summary)
    key_evidence = key_evidence[:4]

    # --- reasoning (deterministic summary) ---
    reasoning_parts = []
    for agent, out in experts.items():
        r = (out.get("reasoning") or "").strip()
        if r:
            reasoning_parts.append(f"{agent}: {r}")
    reasoning = " ".join(reasoning_parts)[:1500]

    return {
        "domain_id": int(domain_id),
        "biological_identity": best_label,
        "biological_identity_conf": float(max(0.0, min(1.0, best_conf))),
        "alternatives": alternatives,
        "margin": float(margin),
        "primary_cell_types": primary_cell_types,
        "key_evidence": key_evidence,
        "reasoning": reasoning,
    }





