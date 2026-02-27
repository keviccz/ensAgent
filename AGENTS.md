# Repository Guidelines

## Project Structure & Module Organization
EnsAgent is organized by pipeline stage. Use these directories as ownership boundaries:
- `ensagent_tools/`: unified tool layer — config management, tool registry, and wrappers for all pipeline stages.
- `Tool-runner/`: clustering orchestration (`orchestrator.py`), tool scripts (`tools/`), post-processing (`postprocess/`), and sample configs (`configs/`).
- `scoring/`: consensus scoring, matrix generation, and annotation entrypoints.
- `ensemble/`: BEST artifact builder (`build_best.py`).
- `annotation/annotation_multiagent/`: multi-agent annotation workflow.
- `streamlit_app/`: UI (`main.py`).
- `ensagent_agent/`: CLI LLM chat agent (`chat.py`).
- `envs/`: R / PY / PY2 environment definitions (single canonical location).
- `tests/`: Python unit tests.
- `endtoend.py`: top-level pipeline runner (loads `pipeline_config.yaml`).
- `pipeline_config.yaml`: user-editable pipeline configuration (copy from `pipeline_config.example.yaml`).

## Build, Test, and Development Commands
- `mamba env create -f environment.yml && mamba activate ensagent`: create the main dev environment.
- `python endtoend.py`: run the full pipeline using `pipeline_config.yaml`.
- `python endtoend.py --data_path "<VISIUM_DIR>" --sample_id "DLPFC_151507"`: override config via CLI.
- `python Tool-runner/orchestrator.py --config Tool-runner/configs/DLPFC_151507.yaml`: run Tool-runner only.
- `python scoring/scoring.py`: run scoring only.
- `streamlit run streamlit_app/main.py`: start the web UI locally.
- `python -m ensagent_agent.chat`: start the CLI LLM agent.
- `python -m unittest discover -s tests -v`: run unit tests.

## Coding Style & Naming Conventions
Target Python 3.10+ and follow PEP 8 with 4-space indentation. Use:
- `snake_case` for files, functions, variables, and CLI flags.
- `PascalCase` for classes (see `tests/` patterns).
- Type hints on new/edited public functions.

No repo-level formatter/linter config is currently committed; keep style consistent with existing modules and run local formatting/linting tools before opening a PR.

## Testing Guidelines
Tests use the standard library `unittest` framework. Place tests in `tests/` and name files `test_*.py`. Keep tests deterministic, fast, and independent of external services. For new functionality, add or update focused tests near the changed behavior (for example orchestration paths, config loading, and health checks). No formal coverage threshold is enforced yet, but critical branches should be exercised.

## Commit & Pull Request Guidelines
- Commit subject in imperative mood, under 72 chars (example: `Add health check for Azure env vars`).
- Keep commits scoped to one logical change.
- PRs should include: purpose, key files changed, commands run for validation, and sample output/screenshots for UI changes (`streamlit_app/`).

## Security & Configuration Tips
Never commit secrets. Configure Azure OpenAI credentials via environment variables (`AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`). Keep `pipeline_config.yaml` git-ignored if it contains API keys. Keep generated outputs under `output/` and avoid checking in large data artifacts unless explicitly required.

## Agent Skills Usage
When a task matches an installed skill (or a user explicitly names one, such as `$data-analysis`), apply that skill workflow first.
- Open the skill's `SKILL.md` and follow only the steps needed for the current task.
- Prefer the minimal set of skills that covers the request; if multiple are needed, state usage order briefly.
- Reuse referenced scripts/templates from the skill directory instead of rewriting from scratch.
- If a skill is unavailable or incomplete, state the gap and continue with the closest safe fallback.
