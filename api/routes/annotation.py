from fastapi import APIRouter
from pathlib import Path
import json
import re

router = APIRouter(prefix="/api/annotation")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _not_found(cluster_id: int, msg: str = "Annotation not yet available. Run Stage D first.") -> dict:
    return {"clusterId": cluster_id, "label": f"Cluster {cluster_id}", "confidence": 0.0,
            "markerGenes": [], "interpretation": msg}


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


def _load_annotations(sample_id: str):
    """Search for domain_annotations.json in known output locations."""
    candidates: list[Path] = []
    for variant in _sample_variants(sample_id):
        candidates.extend(
            [
                _REPO_ROOT / "output" / "best" / variant / "annotation_output" / "domain_annotations.json",
                _REPO_ROOT / "output" / "annotation_runs" / variant / "domain_annotations.json",
                _REPO_ROOT / "output" / "best" / variant / f"{variant}_annotation.json",
            ]
        )
    candidates.append(_REPO_ROOT / "output" / "best" / "annotation_output" / "domain_annotations.json")

    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


@router.get("/dialogue/{sample_id}")
def get_dialogue(sample_id: str):
    """Return Proposer/Critic round logs for the given sample, grouped by domain."""
    rounds: dict = {}  # domain_id -> [round_log, ...]

    for variant in _sample_variants(sample_id):
        # Search annotation_output dirs: output/best/{variant}/annotation_output/{timestamp}/
        for base in [
            _REPO_ROOT / "output" / "best" / variant / "annotation_output",
            _REPO_ROOT / "output" / "annotation_runs" / variant,
        ]:
            if not base.exists():
                continue
            # Each subdirectory is a timestamp run
            run_dirs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
            if not run_dirs:
                # Files directly in base (no timestamp subdir)
                run_dirs = [base]
            for run_dir in run_dirs[:3]:  # latest 3 runs
                for log_file in sorted(run_dir.glob("domain_*_round_*.json")):
                    m = re.match(r"domain_(\d+)_round_(\d+)\.json", log_file.name)
                    if not m:
                        continue
                    domain_id = int(m.group(1))
                    round_idx = int(m.group(2))
                    try:
                        data = json.loads(log_file.read_text(encoding="utf-8"))
                        if domain_id not in rounds:
                            rounds[domain_id] = []
                        rounds[domain_id].append({
                            "round": round_idx,
                            "run_dir": run_dir.name,
                            **data,
                        })
                    except Exception:
                        continue
        if rounds:
            break

    # Sort rounds within each domain
    for d in rounds:
        rounds[d].sort(key=lambda x: x["round"])

    return {
        "sample_id": sample_id,
        "domains": sorted(rounds.keys()),
        "rounds": rounds,
    }


@router.get("/{sample_id}/{cluster_id}")
def get_annotation(sample_id: str, cluster_id: int):
    data = _load_annotations(sample_id)
    if data is None:
        return _not_found(cluster_id)

    # List format: [{domain_id, biological_identity, ...}, ...]
    if isinstance(data, list):
        for item in data:
            if item.get("domain_id") == cluster_id:
                meta = item.get("meta", {})
                return {
                    "clusterId": cluster_id,
                    "label": item.get("biological_identity", f"Domain {cluster_id}"),
                    "confidence": float(item.get("biological_identity_conf", 0.0)),
                    "markerGenes": [e for e in item.get("key_evidence", []) if "Marker:" in e],
                    "interpretation": item.get("function") or item.get("reasoning", "")[:500],
                    "cellTypes": item.get("primary_cell_types", []),
                    "finalScore": float(meta.get("final_score", 0.0)),
                }
        return _not_found(cluster_id, "No annotation found for this cluster.")

    # Dict format: {"1": {...}}
    cluster_key = str(cluster_id)
    if cluster_key not in data:
        return _not_found(cluster_id, "No annotation found for this cluster.")
    ann = data[cluster_key]
    return {
        "clusterId": cluster_id,
        "label":          ann.get("label", f"Cluster {cluster_id}"),
        "confidence":     float(ann.get("confidence", 0.0)),
        "markerGenes":    ann.get("marker_genes", []),
        "interpretation": ann.get("interpretation", ""),
    }
