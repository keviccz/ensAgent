from fastapi import APIRouter, Query
from pathlib import Path
import csv

router = APIRouter(prefix="/api/data")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _sample_variants(sample_id: str) -> list[str]:
    variants: list[str] = []
    sid = str(sample_id or "").strip()
    if not sid:
        return variants
    variants.append(sid)
    if "_" in sid:
        short = sid.split("_", 1)[1]
        if short and short not in variants:
            variants.append(short)
    return variants


def _find_best_spot_csv(sample_id: str) -> Path | None:
    candidates: list[Path] = []
    for variant in _sample_variants(sample_id):
        candidates.append(_REPO_ROOT / "output" / "best" / variant / f"BEST_{variant}_spot.csv")
        candidates.append(_REPO_ROOT / "Tool-runner" / variant / f"BEST_{variant}_spot.csv")
    for base in [_REPO_ROOT / "output" / "best", _REPO_ROOT / "Tool-runner"]:
        if base.exists():
            for p in base.rglob("BEST_*_spot.csv"):
                candidates.append(p)
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_scoring_dir(sample_id: str) -> Path | None:
    candidates: list[Path] = []
    for variant in _sample_variants(sample_id):
        candidates.append(_REPO_ROOT / "scoring" / "output" / variant / "consensus")
    candidates.append(_REPO_ROOT / "scoring" / "output" / "consensus")
    for d in candidates:
        if d.exists() and (d / "scores_matrix.csv").exists() and (d / "labels_matrix.csv").exists():
            return d
    return None


@router.get("/spatial")
def get_spatial(sample_id: str = Query("DLPFC_151507")):
    best_path = _find_best_spot_csv(sample_id)
    if best_path is None:
        return {"spots": []}
    spots = []
    with open(best_path, newline="") as f:
        for row in csv.DictReader(f):
            spots.append({
                "spotId":  row.get("spot_id", ""),
                "x":       float(row.get("imagecol", row.get("x", 0))),
                "y":       float(row.get("imagerow", row.get("y", 0))),
                "cluster": int(float(row.get("spatial_domain", row.get("Ours_domain", row.get("domain", row.get("label", 0)))))),
            })
    return {"spots": spots}


@router.get("/scores")
def get_scores(sample_id: str = Query("DLPFC_151507")):
    scoring_dir = _find_scoring_dir(sample_id)
    if scoring_dir is None:
        return {"rows": []}

    spot_scores: dict = {}
    methods: list = []
    with open(scoring_dir / "scores_matrix.csv", newline="") as f:
        reader = csv.DictReader(f)
        methods = [c for c in (reader.fieldnames or []) if c]
        for row in reader:
            sid = row.get("", "")
            spot_scores[sid] = {m: float(row.get(m, 0) or 0) for m in methods}

    spot_labels: dict = {}
    with open(scoring_dir / "labels_matrix.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("", "")
            spot_labels[sid] = {m: str(row.get(m, "?")) for m in methods}

    from collections import defaultdict
    agg: dict = {m: defaultdict(list) for m in methods}
    for sid, s in spot_scores.items():
        lbl = spot_labels.get(sid, {})
        for m in methods:
            agg[m][lbl.get(m, "?")].append(s.get(m, 0.0))

    rows = []
    for m in methods:
        domain_avgs = {
            d: round(sum(v) / len(v), 4)
            for d, v in sorted(agg[m].items(), key=lambda x: x[0])
            if v
        }
        rows.append({"method": m, "scores": domain_avgs})

    return {"rows": rows}
