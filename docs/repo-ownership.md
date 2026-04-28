# Repository Ownership Map

Updated: 2026-04-10

## Purpose

This document defines the current responsibilities of the main EnsAgent directories so runtime code, historical material, documentation, generated artifacts, and local state do not drift together again.

## Categories

- `Runtime core`: source code or entrypoints used by the active pipeline.
- `Development support`: tests, environment definitions, health checks, and launch helpers.
- `Documentation`: maintained project documentation.
- `Generated artifacts`: runtime results, build outputs, uploads, and caches.
- `Legacy`: code or documents retained only for migration or historical context.

## Directory Ownership

| Path | Category | Responsibility |
| ---- | -------- | -------------- |
| `ensagent_tools/` | Runtime core | Shared tool layer, configuration model, Stage A-D wrappers, and pipeline orchestration. |
| `Tool-runner/` | Runtime core | Stage A clustering orchestration, method scripts, post-processing, and example configs. |
| `scoring/` | Runtime core | Stage B scoring, consensus matrix generation, and visual scoring integration. |
| `ensemble/` | Runtime core | Stage C BEST artifact generation. |
| `annotation/` | Runtime core | Stage D multi-agent annotation. |
| `api/` | Runtime core | FastAPI backend and runtime API routes. |
| `frontend/` | Runtime core | Next.js frontend. |
| `ensagent_agent/` | Runtime core | CLI chat-agent entrypoint. |
| `tests/` | Development support | Unit and runtime-contract regression tests. |
| `tools/` | Development support | Repository health checks and operational helpers. |
| `envs/` | Development support | Conda/Mamba environment definitions. |
| `docs/` | Documentation | Maintained docs, I/O maps, ownership notes, and archived materials. |
| `output/`, `scoring/output/` | Generated artifacts | Runtime outputs; not source-of-truth code. |
| `frontend/.next/` | Generated artifacts | Next.js build output; should stay untracked. |
| `streamlit_app/` | Legacy | Retired UI stack; no longer part of the supported runtime. |

## Single-Entrypoint Rules

- Full pipeline: `python endtoend.py`
- Local web app: `python start.py`
- CLI chat agent: `python -m ensagent_agent.chat`
- Canonical configuration model: `ensagent_tools/config_manager.py`

## Maintenance Rules

- Add runtime code to `ensagent_tools/`, the matching stage directory, `api/`, or `frontend/`.
- Keep `tools/` limited to health checks and operational helpers referenced by docs or tests.
- Add maintained documentation under `docs/`.
- Treat outputs, uploads, screenshots, temporary scripts, and local config as runtime state unless explicitly approved for publication.
