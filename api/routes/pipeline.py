from fastapi import APIRouter
from pydantic import BaseModel
import threading
from typing import Any, Dict

from ensagent_tools import execute_tool
from ensagent_tools.config_manager import load_config as load_pipeline_config
from ensagent_tools.pipeline import stage_toolrunner_outputs

router = APIRouter(prefix="/api/pipeline")

_STAGE_ORDER = ("tool_runner", "scoring", "best", "annotation")
_STAGE_LABELS = {
    "tool_runner": "Tool-Runner",
    "scoring": "Scoring",
    "best": "BEST",
    "annotation": "Annotation",
}
_TOOL_MAP = {
    "tool_runner": "run_tool_runner",
    "scoring": "run_scoring",
    "best": "run_best_builder",
    "annotation": "run_annotation",
}
_STAGE_ALIASES = {
    "tool_runner": "tool_runner",
    "tr": "tool_runner",
    "dp": "tool_runner",
    "scoring": "scoring",
    "sa": "scoring",
    "best": "best",
    "bb": "best",
    "annotation": "annotation",
    "aa": "annotation",
}
_state_lock = threading.Lock()
_runner_thread: threading.Thread | None = None
_stage_state: dict = {
    stage: {"status": "idle", "progress": 0}
    for stage in _STAGE_ORDER
}


def _normalize_stage_name(value: str) -> str:
    return _STAGE_ALIASES.get(str(value or "").strip().lower(), "")


def is_pipeline_running() -> bool:
    return _runner_thread is not None and _runner_thread.is_alive()


def _publish_stage_event(stage: str, status: str, progress: int) -> None:
    try:
        from api.routes.agents import publish_stage_event

        publish_stage_event(stage=stage, status=status, progress=progress)
    except Exception:
        return


def _set_stage(stage: str, *, status: str, progress: int) -> None:
    should_publish = False
    with _state_lock:
        previous = dict(_stage_state[stage])
        _stage_state[stage]["status"] = status
        _stage_state[stage]["progress"] = progress
        should_publish = previous != _stage_state[stage]
    if should_publish:
        _publish_stage_event(stage, status, progress)


def _reset_stage_state(cfg) -> None:
    with _state_lock:
        for stage in _STAGE_ORDER:
            _stage_state[stage] = {"status": "idle", "progress": 0}
        if getattr(cfg, "skip_tool_runner", False):
            _stage_state["tool_runner"] = {"status": "skipped", "progress": 100}
        if getattr(cfg, "skip_scoring", False):
            _stage_state["scoring"] = {"status": "skipped", "progress": 100}
        if not getattr(cfg, "run_best", True):
            _stage_state["best"] = {"status": "skipped", "progress": 100}
        if not getattr(cfg, "run_annotation_multiagent", True):
            _stage_state["annotation"] = {"status": "skipped", "progress": 100}


def _run_pipeline_worker() -> None:
    cfg = load_pipeline_config()
    _reset_stage_state(cfg)

    tool_out = cfg.resolved_tool_output_dir()
    scoring_input = cfg.resolved_scoring_input_dir()
    best_out = cfg.resolved_best_output_dir()

    if not cfg.skip_tool_runner:
        _set_stage("tool_runner", status="running", progress=0)
        result = execute_tool("run_tool_runner", {"output_dir": str(tool_out)}, cfg)
        if not result.get("ok"):
            _set_stage("tool_runner", status="error", progress=0)
            return
        _set_stage("tool_runner", status="done", progress=100)

    if not cfg.skip_scoring:
        if not cfg.skip_tool_runner:
            try:
                stage_toolrunner_outputs(
                    tool_output_dir=tool_out,
                    scoring_input_dir=scoring_input,
                    sample_id=cfg.sample_id,
                    overwrite=cfg.overwrite_staging,
                )
            except Exception:
                _set_stage("scoring", status="error", progress=0)
                return

        _set_stage("scoring", status="running", progress=0)
        result = execute_tool(
            "run_scoring",
            {
                "input_dir": str(scoring_input),
                "output_dir": str(cfg.resolved_scoring_output_dir()),
            },
            cfg,
        )
        if not result.get("ok"):
            _set_stage("scoring", status="error", progress=0)
            return
        _set_stage("scoring", status="done", progress=100)

    if cfg.run_best:
        _set_stage("best", status="running", progress=0)
        result = execute_tool("run_best_builder", {"output_dir": str(best_out)}, cfg)
        if not result.get("ok"):
            _set_stage("best", status="error", progress=0)
            return
        _set_stage("best", status="done", progress=100)

    if cfg.run_annotation_multiagent:
        _set_stage("annotation", status="running", progress=0)
        result = execute_tool(
            "run_annotation",
            {
                "data_dir": str(best_out),
                "output_dir": str(cfg.resolved_annotation_output_dir()),
            },
            cfg,
        )
        if not result.get("ok"):
            _set_stage("annotation", status="error", progress=0)
            return
        _set_stage("annotation", status="done", progress=100)


@router.get("/status")
def get_status():
    with _state_lock:
        stages = [{"name": k, "label": _STAGE_LABELS.get(k, k), **v} for k, v in _stage_state.items()]
    return {"stages": stages, "running": is_pipeline_running()}

@router.post("/run")
def run_pipeline():
    global _runner_thread
    if is_pipeline_running():
        return {"ok": False, "error": "Pipeline is already running"}
    _runner_thread = threading.Thread(target=_run_pipeline_worker, daemon=True)
    _runner_thread.start()
    return {"ok": True}

class StageRequest(BaseModel):
    name: str = ""
    stage: str = ""

@router.post("/stage/{name}")
def run_stage(name: str):
    global _runner_thread
    stage_name = _normalize_stage_name(name)
    if not stage_name:
        return {"ok": False, "error": f"Unknown stage: {name}"}
    if is_pipeline_running():
        return {"ok": False, "error": "Pipeline is already running"}

    cfg = load_pipeline_config()
    tool_out = cfg.resolved_tool_output_dir()
    scoring_input = cfg.resolved_scoring_input_dir()
    best_out = cfg.resolved_best_output_dir()

    def _run():
        _set_stage(stage_name, status="running", progress=0)
        try:
            args: Dict[str, Any] = {}
            if stage_name == "tool_runner":
                args["output_dir"] = str(tool_out)
            if stage_name == "scoring":
                if tool_out.exists():
                    stage_toolrunner_outputs(
                        tool_output_dir=tool_out,
                        scoring_input_dir=scoring_input,
                        sample_id=cfg.sample_id,
                        overwrite=cfg.overwrite_staging,
                    )
                args["input_dir"] = str(scoring_input)
                args["output_dir"] = str(cfg.resolved_scoring_output_dir())
            if stage_name == "best":
                args["output_dir"] = str(best_out)
            if stage_name == "annotation":
                args["data_dir"] = str(best_out)
                args["output_dir"] = str(cfg.resolved_annotation_output_dir())

            result = execute_tool(_TOOL_MAP[stage_name], args, cfg)
            if result.get("ok"):
                _set_stage(stage_name, status="done", progress=100)
            else:
                _set_stage(stage_name, status="error", progress=0)
        except Exception:
            _set_stage(stage_name, status="error", progress=0)

    _runner_thread = threading.Thread(target=_run, daemon=True)
    _runner_thread.start()
    return {"ok": True, "stage": stage_name}

@router.post("/reset")
def reset_stage(req: StageRequest):
    raw_name = getattr(req, "name", "") or getattr(req, "stage", "")
    stage_name = _normalize_stage_name(raw_name)
    if not stage_name:
        return {"ok": False, "error": f"Unknown stage: {raw_name}"}
    with _state_lock:
        if _stage_state[stage_name]["status"] == "running":
            return {"ok": False, "error": f"Stage is running: {stage_name}"}
    _set_stage(stage_name, status="idle", progress=0)
    return {"ok": True, "stage": stage_name}

@router.post("/skip")
def skip_stage(req: StageRequest):
    raw_name = getattr(req, "name", "") or getattr(req, "stage", "")
    stage_name = _normalize_stage_name(raw_name)
    if not stage_name:
        return {"ok": False, "error": f"Unknown stage: {raw_name}"}
    with _state_lock:
        if _stage_state[stage_name]["status"] == "running":
            return {"ok": False, "error": f"Stage is already running: {stage_name}"}
    _set_stage(stage_name, status="skipped", progress=100)
    return {"ok": True, "stage": stage_name}
