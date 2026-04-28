# EnsAgent Development Progress

Last updated: 2026-04-05

This file records the main development changes that led to the current FastAPI + Next.js stack.

## Completed Work

### T1: Streaming Provider Runtime

Files:

- `scoring/provider_runtime.py`

Changes:

- Added `completion_text_stream(config, messages, ...)`.
- Wrapped synchronous LiteLLM streaming in an async generator.
- Fixed the `resolve_provider_config(**str_cfg)` call path for keyword-only arguments.

### T2: Frontend UI Fixes

Files:

- `frontend/components/analysis/ScoresMatrix.tsx`
- `frontend/app/chat/page.tsx`

Changes:

- Fixed method labels in `ScoresMatrix` so method names such as `BASS` are not rendered with an extra `D` prefix.
- Increased low-contrast empty-state text opacity in the chat page.

### T3: Function-Calling Chat Loop

Files:

- `api/routes/chat.py`
- `frontend/lib/types.ts`
- `frontend/lib/store.ts`
- `frontend/lib/api.ts`
- `frontend/app/chat/page.tsx`
- `frontend/components/chat/ChatMessages.tsx`

Backend changes:

- Added LiteLLM tool-calling through `TOOL_SCHEMAS`.
- Added a bounded multi-round tool loop with `MAX_TOOL_ROUNDS=5`.
- Streamed tool-call and tool-result events through SSE.

Frontend changes:

- Added a `ToolCall` result field.
- Added streaming tool-call state to the store.
- Added tool-call and tool-result callbacks to `createChatStream`.
- Rendered in-flight tool calls through `ToolCallBlock`.

### T4: Scores API Data Shape

File:

- `api/routes/data.py`

Changes:

- Reworked `GET /api/data/scores` to return frontend-ready method-by-domain scores.
- Joined score and label matrices before aggregating by method/domain.
- Updated `GET /api/data/spatial` to recognize BEST output columns such as `spatial_domain`.

### T5: Repository Cleanup

Changes:

- Removed temporary debug images and ad-hoc scripts from the root.
- Moved design screenshots into archive documentation.
- Added ignore rules for generated and local-only files.

### T6: Stage B/C Smoke Test

Environment:

- `ens_dev`
- `vlm_off=True`

Observed outputs:

- `scoring/output/consensus/scores_matrix.csv`
- `scoring/output/consensus/labels_matrix.csv`
- `output/best/151507/BEST_151507_spot.csv`
- `output/best/151507/BEST_151507_DEGs.csv`
- `output/best/151507/BEST_151507_PATHWAY.csv`
- `output/best/151507/151507_result.png`

## Known Follow-Ups

| Item | Status | Note |
| ---- | ------ | ---- |
| Complete pic_analyze dependencies in `ens_dev` | Pending | Required for full visual scoring in that environment. |
| Stage D full environment run | Pending | Multi-agent annotation should be tested in a fully provisioned runtime. |
| Frontend analysis page browser verification | Pending | Data display should be checked in a browser session. |
| Chat tool-call UI verification | Pending | ToolCallBlock rendering should be verified with a real tool-triggering prompt. |

## Environments

| Environment | Purpose |
| ----------- | ------- |
| `ens_dev` | FastAPI + Next.js development runtime. |
| `ensagent` | Full production-style runtime. |
| `ensagent_R` | R clustering methods: IRIS, BASS, DR-SC, BayesSpace. |
| `ensagent_PY` | PyTorch clustering methods: SEDR, GraphST, STAGATE. |
| `ensagent_PY2` | TensorFlow/stLearn runtime. |

## Launch Commands

```bash
# Backend and frontend together
python start.py

# Frontend only
cd frontend
npm run dev
```
