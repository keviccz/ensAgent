from fastapi import APIRouter
from pathlib import Path
from pydantic import BaseModel
from api.deps import load_config, save_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FALLBACK_METHODS = ["IRIS", "BASS", "DR-SC", "BayesSpace", "SEDR", "GraphST", "STAGATE", "stLearn"]

router = APIRouter(prefix="/api/config")

@router.get("/load")
def get_config():
    return load_config()

@router.post("/save")
def post_config(payload: dict):
    cfg = load_config()
    cfg.update(payload)
    save_config(cfg)
    return {"ok": True}

class TestRequest(BaseModel):
    api_provider: str = ""
    api_key: str = ""
    api_model: str = ""
    api_endpoint: str = ""
    api_version: str = ""

@router.get("/methods")
def get_methods():
    """Return list of available clustering methods by scanning Tool-runner/tools/."""
    tools_dir = _REPO_ROOT / "Tool-runner" / "tools"
    if not tools_dir.exists():
        return {"methods": _FALLBACK_METHODS}
    methods = []
    for f in sorted(tools_dir.iterdir()):
        if f.suffix.lower() in (".py", ".r") and not f.name.startswith("_"):
            name = f.stem  # e.g. "iris_tool" -> "iris_tool"
            # Map filename to canonical method name
            name_map = {
                "iris_tool": "IRIS", "bass_tool": "BASS",
                "drsc_tool": "DR-SC", "bayesspace_tool": "BayesSpace",
                "sedr_tool": "SEDR", "graphst_tool": "GraphST",
                "stagate_tool": "STAGATE", "stlearn_tool": "stLearn",
            }
            canonical = name_map.get(name.lower(), name)
            methods.append(canonical)
    return {"methods": methods if methods else _FALLBACK_METHODS}


@router.get("/recent_samples")
def get_recent_samples():
    """Return list of sample IDs that have processed data available."""
    samples: list[str] = []
    seen: set[str] = set()

    def _add(s: str):
        if s and s not in seen:
            seen.add(s)
            samples.append(s)

    # Scan output/best/
    best_dir = _REPO_ROOT / "output" / "best"
    if best_dir.exists():
        for d in sorted(best_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                _add(d.name)

    # Scan output/annotation_runs/
    ann_dir = _REPO_ROOT / "output" / "annotation_runs"
    if ann_dir.exists():
        for d in sorted(ann_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                _add(d.name)

    # Scan Tool-runner/ for spot CSVs
    tr_dir = _REPO_ROOT / "Tool-runner"
    if tr_dir.exists():
        for p in tr_dir.rglob("BEST_*_spot.csv"):
            name = p.stem.replace("BEST_", "").replace("_spot", "")
            _add(name)

    # Always include example data sample
    if ((_REPO_ROOT / "example_data" / "domain_annotations.json").exists() or
            (_REPO_ROOT / "example_data" / "scores_matrix.csv").exists()):
        _add("DLPFC_151507")
        _add("151507")

    return {"samples": samples if samples else ["DLPFC_151507"]}


@router.post("/test_connection")
def test_connection(req: TestRequest):
    try:
        import litellm
        litellm.completion(
            model=f"{req.api_provider}/{req.api_model}" if req.api_provider not in ("openai", "") else req.api_model,
            messages=[{"role": "user", "content": "ping"}],
            api_key=req.api_key or None,
            api_base=req.api_endpoint or None,
            api_version=req.api_version or None,
            max_tokens=1,
        )
        return {"ok": True, "message": "Connection successful"}
    except Exception as e:
        return {"ok": False, "message": str(e)}
