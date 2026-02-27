import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=int, required=True)
    ap.add_argument(
        "--ann_json",
        type=str,
        default="ARI&Picture_DLPFC_151507/annotation_output/domain_annotations.json",
    )
    args = ap.parse_args()

    ann = _load_json(Path(args.ann_json))
    rec = None
    for it in ann:
        if int(it.get("domain_id") or 0) == int(args.domain):
            rec = it
            break
    if rec is None:
        raise SystemExit(f"Domain {args.domain} not found in {args.ann_json}")

    meta = rec.get("meta") or {}
    log_path = str(meta.get("log_path") or "")
    log_file = _resolve_log_path(log_path)
    if log_file is None:
        raise SystemExit(f"Could not resolve log_path: {log_path}")

    folder = log_file.parent
    files = sorted(folder.glob(f"domain_{int(args.domain)}_round_*.json"))
    print(f"domain={args.domain}  folder={folder}  rounds={len(files)}")
    print("-" * 90)
    for f in files:
        obj = _load_json(f)
        run_info = obj.get("run_info") or {}
        proposer = (obj.get("proposer") or {}).get("annotation") or {}
        critic = obj.get("critic") or {}
        gate = obj.get("gate") or {}
        print(
            f"r{run_info.get('round')}: label={proposer.get('biological_identity')} "
            f"conf={proposer.get('biological_identity_conf')} "
            f"critic={critic.get('critic_score')} final={gate.get('final_score')} "
            f"passed={gate.get('passed')} fail={gate.get('fail_reasons')} "
            f"planned={run_info.get('loop_planned_reason','')}"
        )


if __name__ == '__main__':
    main()

















