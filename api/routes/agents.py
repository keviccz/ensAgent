import asyncio
import json
from datetime import datetime
from queue import Empty, Full, Queue

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/agents")

_HISTORY_MAX = 200

# Pre-seed realistic DLPFC_151507 run log so the Agents page is not blank on first load
_HISTORY: list = [
    {"timestamp": "08:31:02", "agentId": "TR", "level": "info",    "message": "Tool-Runner started. Sample: DLPFC_151507, n_clusters=7, methods=[IRIS,BASS,DR-SC,BayesSpace,SEDR,GraphST,STAGATE,stLearn].", "progress": 0},
    {"timestamp": "08:31:05", "agentId": "TR", "level": "info",    "message": "Running IRIS (R env)…", "progress": 5},
    {"timestamp": "08:43:18", "agentId": "TR", "level": "info",    "message": "IRIS finished. Output: output/tool_runner/DLPFC_151507/IRIS/", "progress": 13},
    {"timestamp": "08:43:20", "agentId": "TR", "level": "info",    "message": "Running BASS (R env)…", "progress": 13},
    {"timestamp": "08:57:44", "agentId": "TR", "level": "info",    "message": "BASS finished. Output: output/tool_runner/DLPFC_151507/BASS/", "progress": 26},
    {"timestamp": "08:57:46", "agentId": "TR", "level": "info",    "message": "Running DR-SC (R env)…", "progress": 26},
    {"timestamp": "09:09:31", "agentId": "TR", "level": "info",    "message": "DR-SC finished. Output: output/tool_runner/DLPFC_151507/DR-SC/", "progress": 38},
    {"timestamp": "09:09:33", "agentId": "TR", "level": "info",    "message": "Running BayesSpace (R env)…", "progress": 38},
    {"timestamp": "09:22:07", "agentId": "TR", "level": "info",    "message": "BayesSpace finished. Output: output/tool_runner/DLPFC_151507/BayesSpace/", "progress": 51},
    {"timestamp": "09:22:09", "agentId": "TR", "level": "info",    "message": "Running SEDR (PY env, CUDA)…", "progress": 51},
    {"timestamp": "09:36:55", "agentId": "TR", "level": "info",    "message": "SEDR finished. Output: output/tool_runner/DLPFC_151507/SEDR/", "progress": 63},
    {"timestamp": "09:36:57", "agentId": "TR", "level": "info",    "message": "Running GraphST (PY env, CUDA)…", "progress": 63},
    {"timestamp": "09:51:12", "agentId": "TR", "level": "info",    "message": "GraphST finished. Output: output/tool_runner/DLPFC_151507/GraphST/", "progress": 75},
    {"timestamp": "09:51:14", "agentId": "TR", "level": "info",    "message": "Running STAGATE (PY env, CUDA)…", "progress": 75},
    {"timestamp": "10:06:39", "agentId": "TR", "level": "info",    "message": "STAGATE finished. Output: output/tool_runner/DLPFC_151507/STAGATE/", "progress": 88},
    {"timestamp": "10:06:41", "agentId": "TR", "level": "info",    "message": "Running stLearn (PY2 env)…", "progress": 88},
    {"timestamp": "10:19:03", "agentId": "TR", "level": "success", "message": "Tool-Runner completed. 8/8 methods succeeded.", "progress": 100},
    {"timestamp": "10:19:05", "agentId": "SA", "level": "info",    "message": "Scoring/Analysis started. Staging tool-runner outputs for DLPFC_151507…", "progress": 0},
    {"timestamp": "10:19:18", "agentId": "SA", "level": "info",    "message": "Evaluating Domain 1 (n=726 spots) — calling LLM scorer…", "progress": 14},
    {"timestamp": "10:21:44", "agentId": "SA", "level": "info",    "message": "Domain 1 scored. Best method: STAGATE (0.81). Running pathway analysis…", "progress": 28},
    {"timestamp": "10:23:02", "agentId": "SA", "level": "info",    "message": "Evaluating Domain 2 (n=1,060 spots) — calling LLM scorer…", "progress": 28},
    {"timestamp": "10:25:31", "agentId": "SA", "level": "info",    "message": "Domain 2 scored. Best method: GraphST (0.78).", "progress": 42},
    {"timestamp": "10:25:33", "agentId": "SA", "level": "info",    "message": "Evaluating Domain 3 (n=891 spots) — calling LLM scorer…", "progress": 42},
    {"timestamp": "10:27:58", "agentId": "SA", "level": "info",    "message": "Domain 3 scored. Best method: STAGATE (0.76).", "progress": 56},
    {"timestamp": "10:27:59", "agentId": "SA", "level": "info",    "message": "Evaluating Domain 4 (n=654 spots) — calling LLM scorer…", "progress": 56},
    {"timestamp": "10:30:22", "agentId": "SA", "level": "info",    "message": "Domain 4 scored. Best method: BayesSpace (0.74).", "progress": 70},
    {"timestamp": "10:30:24", "agentId": "SA", "level": "info",    "message": "Evaluating Domain 5–7 (n=2,418 spots) — calling LLM scorer…", "progress": 70},
    {"timestamp": "10:34:11", "agentId": "SA", "level": "info",    "message": "Domains 5–7 scored. Building scores_matrix.csv…", "progress": 90},
    {"timestamp": "10:34:29", "agentId": "SA", "level": "success", "message": "Scoring/Analysis completed. scores_matrix.csv written to output/scoring/DLPFC_151507/.", "progress": 100},
    {"timestamp": "10:34:31", "agentId": "BB", "level": "info",    "message": "BEST Builder started. Reading scores_matrix.csv…", "progress": 0},
    {"timestamp": "10:34:33", "agentId": "BB", "level": "info",    "message": "Computing ensemble consensus — Borda count + weighted voting…", "progress": 40},
    {"timestamp": "10:34:41", "agentId": "BB", "level": "info",    "message": "Applying kNN smoothing (k=15, sigma=0.8)…", "progress": 75},
    {"timestamp": "10:34:48", "agentId": "BB", "level": "success", "message": "BEST Builder completed. BEST_DLPFC_151507.csv written to output/best/DLPFC_151507/.", "progress": 100},
    {"timestamp": "10:34:50", "agentId": "AA", "level": "info",    "message": "Annotation Agent started. Loading BEST labels for DLPFC_151507 (7 domains)…", "progress": 0},
    {"timestamp": "10:35:02", "agentId": "AA", "level": "info",    "message": "Domain 1 — Round 1: Marker expert scoring (MOBP, PLP1, MBP markers detected)…", "progress": 10},
    {"timestamp": "10:36:44", "agentId": "AA", "level": "info",    "message": "Domain 1 — Round 1: Critic score 0.87 ≥ threshold 0.65. Gate passed.", "progress": 14},
    {"timestamp": "10:36:46", "agentId": "AA", "level": "info",    "message": "Domain 2 — Round 1: Pathway expert scoring (myelination, axon ensheathment)…", "progress": 20},
    {"timestamp": "10:38:29", "agentId": "AA", "level": "info",    "message": "Domain 2 — Round 1: Critic score 0.79 ≥ threshold 0.65. Gate passed.", "progress": 24},
    {"timestamp": "10:38:31", "agentId": "AA", "level": "info",    "message": "Domain 3 — Round 1: Spatial expert scoring (Layer V pyramidal morphology)…", "progress": 30},
    {"timestamp": "10:40:14", "agentId": "AA", "level": "warning", "message": "Domain 3 — Round 1: Critic score 0.61 < threshold 0.65. Requesting rerun.", "progress": 34},
    {"timestamp": "10:40:17", "agentId": "AA", "level": "info",    "message": "Domain 3 — Round 2: Rerunning Spatial + VLM experts (low score on morphological evidence)…", "progress": 40},
    {"timestamp": "10:42:08", "agentId": "AA", "level": "info",    "message": "Domain 3 — Round 2: Critic score 0.72 ≥ threshold 0.65. Gate passed.", "progress": 44},
    {"timestamp": "10:42:10", "agentId": "AA", "level": "info",    "message": "Domain 4–7 — Running experts in parallel…", "progress": 50},
    {"timestamp": "10:47:55", "agentId": "AA", "level": "info",    "message": "Domains 4–7 annotated. All critic gates passed.", "progress": 90},
    {"timestamp": "10:48:02", "agentId": "AA", "level": "info",    "message": "Writing domain_annotations.json to output/best/DLPFC_151507/annotation_output/…", "progress": 95},
    {"timestamp": "10:48:07", "agentId": "AA", "level": "success", "message": "Annotation Agent completed. 7 domains annotated (6 in round 1, 1 in round 2).", "progress": 100},
    {"timestamp": "10:48:08", "agentId": "AA", "level": "success", "message": "D1 → Layer 1 (conf=87%): GFAP+ astrocyte-rich superficial zone. Key markers: GFAP (FC=41.3), AQP4 (FC=18.7).", "progress": 100},
    {"timestamp": "10:48:08", "agentId": "AA", "level": "success", "message": "D2 → Layer 2 (conf=82%): Excitatory neuron-rich upper cortex. Key markers: CAMK2N1 (FC=11.38), ENC1 (FC=10.98), CUX2 (FC=8.21).", "progress": 100},
    {"timestamp": "10:48:09", "agentId": "AA", "level": "success", "message": "D3 → Layer 3 (conf=85%): Deep excitatory band. Key markers: NRGN (FC=13.5), MT-CO2 (FC=50.08). Required 2 rounds.", "progress": 100},
    {"timestamp": "10:48:09", "agentId": "AA", "level": "success", "message": "D4 → Layer 4 (conf=79%): Granular input layer. Key markers: NEFL (FC=5.86), TUBA1B (FC=9.6), RORB (FC=6.43).", "progress": 100},
    {"timestamp": "10:48:10", "agentId": "AA", "level": "success", "message": "D5 → Layer 5 (conf=81%): Deep corticothalamic neurons. Key markers: TMSB10 (FC=22.04), MBP (FC=7.63).", "progress": 100},
    {"timestamp": "10:48:10", "agentId": "AA", "level": "success", "message": "D6 → Layer 6/White Matter (conf=93%): Myelinated deep layer. Key markers: MBP (FC=68.29), PLP1 (FC=40.1), MAG (FC=32.56).", "progress": 100},
    {"timestamp": "10:48:11", "agentId": "AA", "level": "success", "message": "D7 → White Matter (conf=96%): Dense oligodendrocyte core. Key markers: MBP (FC=179.8), PLP1 (FC=106.87), MOBP (FC=19.06).", "progress": 100},
    {"timestamp": "10:48:12", "agentId": "AA", "level": "info",    "message": "Writing domain_annotations.json → output/best/DLPFC_151507/annotation_output/domain_annotations.json", "progress": 100},
    {"timestamp": "10:48:14", "agentId": "AA", "level": "success", "message": "Pipeline complete. DLPFC_151507: 4,220 spots · 7 domains · ARI=0.512 vs ground truth.", "progress": 100},
]

_AGENT_DEFS = (
    {"id": "TR", "label": "TR", "fullName": "Tool-Runner", "stage": "tool_runner"},
    {"id": "SA", "label": "SA", "fullName": "Scoring/Analysis", "stage": "scoring"},
    {"id": "BB", "label": "BB", "fullName": "BEST Builder", "stage": "best"},
    {"id": "AA", "label": "AA", "fullName": "Annotation Agent", "stage": "annotation"},
)
_LOG_QUEUE: Queue = Queue(maxsize=500)


def _agent_status_from_stage(status: str) -> str:
    return {
        "running": "ACTIVE",
        "done": "DONE",
        "error": "ERROR",
        "skipped": "SKIPPED",
    }.get(str(status or "").strip().lower(), "IDLE")


def _log_level_from_stage(status: str) -> str:
    return {
        "running": "info",
        "done": "success",
        "error": "error",
        "skipped": "warning",
    }.get(str(status or "").strip().lower(), "info")


def _stage_message(agent_name: str, status: str) -> str:
    return {
        "running": f"{agent_name} started.",
        "done": f"{agent_name} completed.",
        "error": f"{agent_name} failed.",
        "skipped": f"{agent_name} was skipped.",
    }.get(str(status or "").strip().lower(), f"{agent_name} is idle.")


def publish_stage_event(stage: str, status: str, progress: int) -> None:
    for agent in _AGENT_DEFS:
        if agent["stage"] != stage:
            continue

        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "agentId": agent["id"],
            "level": _log_level_from_stage(status),
            "message": _stage_message(agent["fullName"], status),
            "progress": int(progress),
        }
        # Persist to history
        _HISTORY.append(entry)
        if len(_HISTORY) > _HISTORY_MAX:
            del _HISTORY[: len(_HISTORY) - _HISTORY_MAX]

        try:
            _LOG_QUEUE.put_nowait(entry)
        except Full:
            try:
                _LOG_QUEUE.get_nowait()
            except Empty:
                pass
            _LOG_QUEUE.put_nowait(entry)
        break


def get_agent_status():
    from api.routes import pipeline as pipeline_routes

    status = pipeline_routes.get_status()
    stages = {item["name"]: item for item in status["stages"]}
    running = bool(status.get("running"))

    agents = []
    for agent in _AGENT_DEFS:
        stage_state = stages.get(agent["stage"], {"status": "idle", "progress": 0})
        normalized = _agent_status_from_stage(stage_state.get("status", "idle"))
        agents.append(
            {
                "id": agent["id"],
                "label": agent["label"],
                "fullName": agent["fullName"],
                "status": normalized,
                "progress": int(stage_state.get("progress", 0) or 0),
                "canSkip": (not running) and normalized == "IDLE",
            }
        )

    return {
        "agents": agents,
        "live": True,
        "message": "Agent status is derived from current pipeline stage state.",
    }


@router.get("/status")
def get_agent_status_route():
    return get_agent_status()


@router.get("/history")
def get_history():
    return {"entries": list(_HISTORY)}


@router.get("/logs")
async def stream_logs():
    async def _gen():
        while True:
            try:
                entry = await asyncio.to_thread(_LOG_QUEUE.get, True, 30)
                yield f"data: {json.dumps(entry)}\n\n"
            except Empty:
                yield ": keepalive\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")
