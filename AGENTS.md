# AGENTS.md — EnsAgent Agent Guide

Use this file as the primary onboarding document for coding agents working in this repository. Open `README.md` for fuller product/setup context and `CLAUDE.md` for additional project notes, but start here.

## Repo Purpose
EnsAgent is an ensemble multi-agent framework for spatial transcriptomics analysis on 10x Visium data. The stable pipeline is:

1. Tool-Runner
2. Scoring
3. BEST Builder
4. Annotation

## Architecture Map
- `ensagent_tools/`: shared integration layer. Holds config loading/saving, tool schemas, tool dispatch, subprocess helpers, and stage wrappers used by the CLI agent, API, and pipeline entrypoints.
- `Tool-runner/`: Stage A clustering orchestration and post-processing.
- `scoring/`: Stage B scoring, consensus matrix generation, provider runtime, and annotation entrypoints.
- `ensemble/`: Stage C BEST artifact generation via `build_best.py`.
- `annotation/annotation_multiagent/`: Stage D multi-agent annotation workflow.
- `api/`: FastAPI backend. Route modules live under `api/routes/`.
- `frontend/`: Next.js web app. App routes are under `frontend/app/`, shared UI under `frontend/components/`, and client API/state types under `frontend/lib/`.
- `ensagent_agent/`: CLI chat agent that uses the same tool registry as the web stack.
- `envs/`: canonical environment YAMLs for R / PY / PY2 tool environments.
- `tests/`: `unittest` test suite.
- `start.py`: launches the FastAPI backend and Next.js frontend together for local development.
- `endtoend.py`: top-level pipeline runner that loads `pipeline_config.yaml`.
- `pipeline_config.example.yaml`: template for local pipeline configuration.
- `pipeline_config.yaml`: local working config copied from the example template.

## Task Routing
- Pipeline behavior or stage orchestration:
  Start in `ensagent_tools/`, then inspect the stage implementation in `Tool-runner/`, `scoring/`, `ensemble/`, or `annotation/`.
- API behavior:
  Start in `api/routes/` and check the matching wrapper or config logic in `ensagent_tools/`.
- Frontend behavior:
  Start in `frontend/app/` for route-level changes, `frontend/components/` for UI behavior, and `frontend/lib/` for API clients, types, and store logic.
- Chat or tool-calling behavior:
  Start in `api/routes/chat.py`, `ensagent_tools/registry.py`, and `ensagent_agent/chat.py`.
- Config loading or persistence:
  Start in `ensagent_tools/config_manager.py`, then verify related defaults and comments in `pipeline_config.example.yaml`.
- Environment setup and health checks:
  Start in `envs/`, `ensagent_tools/env_manager.py`, and `tools/health_check.py`.

## Core Commands
- Create the main environment and install dependencies:
  `mamba env create -f environment.yml && mamba activate ensagent`
  `python -m pip install -r requirements.txt`
- Run the full pipeline:
  `python endtoend.py`
- Run the full pipeline with CLI overrides:
  `python endtoend.py --data_path "<VISIUM_DIR>" --sample_id "DLPFC_151507"`
- Run Tool-Runner only:
  `python Tool-runner/orchestrator.py --config Tool-runner/configs/DLPFC_151507.yaml`
- Run Scoring only:
  `python scoring/scoring.py`
- Run the local web app stack:
  `python start.py`
- Run the CLI chat agent:
  `python -m ensagent_agent.chat`
- Run the health check:
  `python tools/health_check.py`
- Run Python tests:
  `python -m unittest discover -s tests -v`
- Build the frontend:
  `cd frontend && npm run build`

## Working Rules
- Target Python 3.10+ and follow PEP 8 with 4-space indentation.
- Use `snake_case` for files, functions, variables, and CLI flags. Use `PascalCase` for classes.
- Add type hints to new or edited public Python functions.
- Tests use the standard library `unittest` framework. Add focused tests under `tests/test_*.py`.
- Prefer small, local edits in stable source directories over broad refactors.
- Treat `pipeline_config.yaml` as local user state. Do not edit it unless the task is explicitly about runtime configuration.
- Treat `output/`, `output_log/`, `scoring/output/`, `uploads/`, and `frontend/.next/` as generated or runtime state, not primary sources of truth.
- Do not commit secrets. Prefer environment variables for provider credentials.
- Do not revert unrelated user changes in the worktree.

## Agent Skill Usage
- If a task matches an installed skill, apply that skill workflow first.
- Use the minimal set of skills that covers the task and state the order briefly when multiple skills apply.
- Reuse referenced scripts, templates, and helper assets from the skill when available instead of recreating them.
- If a required skill is unavailable or incomplete, state the gap and continue with the closest safe fallback.

## Validation Guide
- Docs-only or guidance-only changes:
  Re-read the referenced paths and commands for accuracy. Run the narrowest relevant consistency checks if the docs describe tested behavior.
- `ensagent_tools/` or pipeline wrapper changes:
  Run the most relevant `unittest` modules first. Run `python -m unittest discover -s tests -v` if shared behavior changed.
- API changes:
  Run the relevant unit tests if present, then run `python tools/health_check.py`. Use `python start.py` for a local integration smoke test when request/response behavior changed.
- Frontend changes:
  Run `cd frontend && npm run build`. If the change crosses the API/UI boundary, also smoke test with `python start.py`.
- End-to-end workflow changes:
  Validate the smallest relevant stage command before running `python endtoend.py`.

## Commit And PR Expectations
- Commit messages should use the imperative mood and stay under 72 characters.
- Keep each commit scoped to one logical change.
- PRs should include purpose, key files changed, validation commands run, and screenshots for UI changes under `frontend/` or API/UI flow changes under `api/`.

## Security And Configuration Notes
- Preferred credential path:
  `ENSAGENT_API_PROVIDER`, `ENSAGENT_API_KEY`, `ENSAGENT_API_MODEL`, and related environment variables.
- Azure compatibility aliases are still supported:
  `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`.
- Keep large generated data artifacts out of git unless the task explicitly requires them.
