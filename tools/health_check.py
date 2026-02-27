from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _status_from_checks(checks: List[Dict[str, str]]) -> str:
    if any(c["status"] == "fail" for c in checks):
        return "fail"
    if any(c["status"] == "warn" for c in checks):
        return "warn"
    return "pass"


def gather_health_report() -> Dict[str, object]:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    checks: List[Dict[str, str]] = []

    py_ok = sys.version_info >= (3, 10)
    checks.append(
        {
            "name": "python_version",
            "status": "pass" if py_ok else "fail",
            "detail": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
    )

    required_paths = [
        root / "Tool-runner" / "orchestrator.py",
        root / "scoring" / "scoring.py",
        root / "ensemble" / "build_best.py",
        root / "annotation" / "annotation_multiagent" / "orchestrator.py",
        root / "streamlit_app" / "main.py",
    ]
    missing = [str(p.relative_to(root)) for p in required_paths if not p.exists()]
    checks.append(
        {
            "name": "repo_layout",
            "status": "pass" if not missing else "fail",
            "detail": "all required paths exist" if not missing else f"missing: {', '.join(missing)}",
        }
    )

    required_envs = [
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
    ]
    missing_env = [k for k in required_envs if not os.getenv(k)]
    checks.append(
        {
            "name": "azure_openai_env",
            "status": "pass" if not missing_env else "warn",
            "detail": "configured" if not missing_env else f"missing: {', '.join(missing_env)}",
        }
    )

    core_modules = ["pandas", "numpy", "sklearn", "openai", "streamlit"]
    missing_modules = [m for m in core_modules if importlib.util.find_spec(m) is None]
    checks.append(
        {
            "name": "core_modules",
            "status": "pass" if not missing_modules else "warn",
            "detail": "available" if not missing_modules else f"missing: {', '.join(missing_modules)}",
        }
    )

    try:
        importlib.import_module("scoring.config")
        checks.append(
            {
                "name": "scoring_config_import",
                "status": "pass",
                "detail": "import ok",
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "scoring_config_import",
                "status": "fail",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )

    status = _status_from_checks(checks)
    return {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "checks": checks,
    }


def _print_human(report: Dict[str, object]) -> None:
    print(f"Overall: {report['status']}")
    for item in report["checks"]:
        print(f"[{item['status']}] {item['name']}: {item['detail']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EnsAgent local health check")
    ap.add_argument("--json", action="store_true", help="Print JSON report")
    args = ap.parse_args()

    report = gather_health_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report)

    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
