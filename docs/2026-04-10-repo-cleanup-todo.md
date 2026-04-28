# Repository Cleanup Checklist

Date: 2026-04-10

## Purpose

This checklist tracks the cleanup work required to keep EnsAgent publishable and maintainable:

- Keep the active runtime stack clear.
- Separate source code from generated artifacts and local state.
- Document stage-level inputs and outputs.
- Remove or archive legacy Streamlit material.
- Keep API/frontend reads sample-scoped.

## Active Runtime Skeleton

The following paths are part of the active workflow:

- `ensagent_tools/`
- `Tool-runner/`
- `scoring/`
- `ensemble/`
- `annotation/`
- `api/`
- `frontend/`
- `ensagent_agent/`
- `tests/`
- `tools/health_check.py`
- `start.py`
- `endtoend.py`
- `pipeline_config.example.yaml`

## Cleanup Outcomes

- [x] Streamlit UI removed from the supported runtime.
- [x] Active web stack normalized to FastAPI + Next.js.
- [x] Runtime outputs treated as generated artifacts.
- [x] Repository ownership documented in `docs/repo-ownership.md`.
- [x] Pipeline I/O documented in `docs/pipeline-io-map.md`.
- [x] API routes updated to prefer sample-scoped outputs.
- [x] Legacy design and planning documents moved under `docs/archive/`.
- [x] README updated to describe the current public workflow.

## Configuration Cleanup

- [x] Inventory config readers in `api/deps.py`, `api/routes/config.py`, `ensagent_tools/config_manager.py`, stage wrappers, and `endtoend.py`.
- [x] Use `ensagent_tools/config_manager.py` as the canonical model.
- [x] Treat `pipeline_config.example.yaml` as the versioned template.
- [x] Treat `pipeline_config.yaml` as local runtime state.

## API and Output Rules

- [x] `/api/data/scores` reads sample-specific consensus outputs.
- [x] `/api/data/spatial` reads BEST sample outputs.
- [x] `/api/annotation/{sample}/{cluster}` reads sample-specific annotation outputs first.
- [x] Shared output fallbacks are legacy compatibility paths only.

## Completion Criteria

The cleanup is considered complete when:

- Every main directory has a clear responsibility.
- Each stage input/output contract is documented.
- API routes avoid ambiguous shared output paths.
- Local configuration and generated artifacts are not published.
- Legacy UI code is removed or archived.
- The narrow README tests, full unit suite, and frontend build pass.
