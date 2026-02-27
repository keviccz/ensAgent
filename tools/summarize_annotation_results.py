import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _try_load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_log_path(log_path: str) -> Optional[Path]:
    if not log_path:
        return None
    lp = str(log_path).replace("\\\\", "/").replace("\\", "/")
    cands = [
        Path(lp),
        Path("Score Rag") / lp,
        Path(lp.replace("output/annotation_runs", "Score Rag/output/annotation_runs")),
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def _count_rounds(log_file: Path, domain_id: int) -> int:
    parent = log_file.parent
    return len(list(parent.glob(f"domain_{int(domain_id)}_round_*.json")))


def main() -> None:
    ann_path = Path("ARI&Picture_DLPFC_151507/annotation_output/domain_annotations.json")
    if not ann_path.exists():
        raise SystemExit(f"Not found: {ann_path}")
    obj: List[Dict[str, Any]] = _try_load_json(ann_path)

    rows: List[Tuple[int, str, float, bool, float, float, float, List[str], int, str, List[str]]] = []
    for it in obj:
        dom = int(it.get("domain_id") or 0)
        meta = it.get("meta") or {}
        log_path = str(meta.get("log_path") or "")
        log_file = _resolve_log_path(log_path)
        rounds = _count_rounds(log_file, dom) if log_file else 0
        rows.append(
            (
                dom,
                str(it.get("biological_identity")),
                float(it.get("biological_identity_conf") or 0.0),
                bool(meta.get("passed")),
                float(meta.get("final_score") or 0.0),
                float(meta.get("critic_score") or 0.0),
                float(meta.get("peer_score_weighted") or 0.0),
                list(meta.get("fail_reasons") or []),
                rounds,
                log_path,
                list(it.get("primary_cell_types") or []),
            )
        )

    rows.sort(key=lambda x: x[0])
    print(f"N={len(rows)}  file={ann_path}")
    failed = [r[0] for r in rows if not r[3]]
    print(f"failed_domains={failed}")
    print("-" * 80)
    for dom, label, conf, passed, final, critic, peer_w, fail, rounds, log_path, cts in rows:
        print(
            f"D{dom}: label={label} conf={conf:.3f} final={final:.4f} critic={critic:.4f} peer_w={peer_w:.4f} "
            f"passed={passed} rounds={rounds} fail={fail}"
        )
        if cts:
            print(f"  primary_cell_types: {cts}")
        if log_path:
            print(f"  log_path: {log_path}")


if __name__ == "__main__":
    main()

















