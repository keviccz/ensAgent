# CLAUDE.md — EnsAgent

EnsAgent is an ensemble multi-agent framework for spatial transcriptomics analysis. It integrates eight spatial clustering methods, LLM-based evaluation, consensus label selection, and multi-agent domain annotation into a single reproducible pipeline targeting 10x Visium data.

---

## Essential Commands

```bash
# Create main environment and install deps
mamba env create -f environment.yml && mamba activate ensagent
python -m pip install -r requirements.txt

# Run full pipeline (reads pipeline_config.yaml)
python endtoend.py

# Override a single parameter via CLI
python endtoend.py --n_clusters 5 --data_path "Tool-runner/151507"

# Run individual stages
python Tool-runner/orchestrator.py --config Tool-runner/configs/DLPFC_151507.yaml   # Stage A
cd scoring && python scoring.py                                                       # Stage B
python ensemble/build_best.py --sample_id DLPFC_151507 ...                           # Stage C
python scoring/scoring.py --annotation_multiagent ...                                 # Stage D

# Web app
python start.py

# CLI chat agent
python -m ensagent_agent.chat

# Tests
python -m unittest discover -s tests -v

# Health check
python tools/health_check.py
```

---

## Architecture

### Four-Stage Pipeline: A → B → C → D

| Stage | Module | Purpose |
|-------|--------|---------|
| **A** Tool-Runner | `Tool-runner/orchestrator.py` | Run 8 clustering methods in isolated conda envs |
| **B** Scoring | `scoring/scoring.py` | LLM-driven per-domain evaluation; build `scores_matrix.csv` and `labels_matrix.csv` |
| **C** BEST Builder | `ensemble/build_best.py` | Select optimal domain labels; produce `BEST_*` files with optional kNN smoothing |
| **D** Annotation | `annotation/annotation_multiagent/orchestrator.py` | Proposer–Expert–Critic multi-agent annotation loop |

### Directory Ownership

```
ensagent_tools/        # Unified tool layer: config, registry, stage wrappers, subprocess streaming
Tool-runner/           # Stage A: orchestrator + 8 clustering scripts (4 R, 4 Py) + postprocess/
scoring/               # Stage B: LLM scoring, spatial metrics, pathway analysis, visual scoring
ensemble/              # Stage C: BEST label selection (build_best.py)
annotation/            # Stage D: multi-agent annotation workflow
ensagent_agent/        # CLI LLM chat agent (chat.py)
api/                   # FastAPI backend
frontend/              # Next.js web UI
start.py               # Local launcher for FastAPI + Next.js
envs/                  # R / PY / PY2 environment YAML definitions (canonical location)
tests/                 # 20 unit tests (unittest framework)
tools/                 # Utilities: health_check.py, kill_port.py
docs/                  # Design screenshots, plans, specs
endtoend.py            # Top-level CLI pipeline entry point
```

### Conda Environments

| Name | YAML | Clustering methods |
|------|------|--------------------|
| `ensagent` | `environment.yml` | Main env — LLM libs, FastAPI/Next.js support, all tooling |
| `ensagent_R` | `envs/R_environment.yml` | IRIS, BASS, DR-SC, BayesSpace (R 4.4+ / rpy2) |
| `ensagent_PY` | `envs/PY_environment.yml` | SEDR, GraphST, STAGATE (PyTorch 2.0) |
| `ensagent_PY2` | `envs/PY2_environment.yml` | stLearn (TensorFlow 2.11) |

### Key Abstractions

- **`PipelineConfig`** (`ensagent_tools/config_manager.py`): top-level dataclass; loaded from `pipeline_config.yaml` with CLI overrides.
- **`TOOL_REGISTRY` / `TOOL_SCHEMAS`** (`ensagent_tools/registry.py`): dispatch table + OpenAI function-calling schemas for all pipeline tools.
- **`run_subprocess_streaming()`** (`ensagent_tools/subprocess_stream.py`): runs subprocesses with live progress callbacks, cancellation support, and structured events.
- **`resolve_provider_config()` / `completion_text()`** (`scoring/provider_runtime.py`): unified LLM abstraction over 12+ providers via litellm with SDK fallbacks.
- **`DomainEvaluator`** (`scoring/domain_evaluator.py`): per-domain LLM scoring with spatial metrics, DEG/pathway analysis, and visual score integration.
- **`GPTDomainScorer`** (`scoring/gpt_scorer.py`): sends a detailed biological rubric (~340-line `RUBRIC_TEXT`) to the LLM; applies programmatic penalties post-response.
- **`execute_tool()`** (`ensagent_tools/registry.py`): universal tool entry point — CLI agent, Web API, and pipeline all dispatch through this function.

### Web App Architecture

**FastAPI Backend** (`api/`):

| Route module | Endpoints | Purpose |
|-------------|-----------|---------|
| `routes/config.py` | GET/POST `/api/config/*` | Config CRUD + LLM connection test |
| `routes/pipeline.py` | POST `/api/pipeline/{stage}`, GET `/api/pipeline/status` | Background stage execution + real-time status |
| `routes/chat.py` | POST `/api/chat` (SSE) | Streaming tool-calling agent loop (max 5 rounds) |
| `routes/agents.py` | GET `/api/agents/status`, GET `/api/agents/logs` (SSE) | Agent status + activity log stream |
| `routes/data.py` | GET `/api/data/spatial`, GET `/api/data/scores` | Spatial & scores matrix data |
| `routes/annotation.py` | GET `/api/annotation/{sample_id}/{cluster_id}` | Domain annotation retrieval |

- `api/deps.py` — Config loading/saving with `api_*` ↔ `azure_*` alias syncing and type coercion.

**Next.js Frontend** (`frontend/`):

- Stack: Next.js 14 + React 18 + Zustand + Tailwind CSS + Recharts + Lucide icons
- Pages: `chat/` (tool-calling chat), `agents/` (agent dashboard), `analysis/` (spatial visualization), `settings/` (config UI)
- 21 components across `components/chat/`, `components/agents/`, `components/analysis/`, `components/settings/`, `components/layout/`
- `lib/store.ts` — Zustand global state (conversations, pipeline, agents, config)
- `lib/api.ts` — Fetch client for all FastAPI endpoints
- `lib/types.ts` — TypeScript interfaces for all API contracts

---

## Configuration

### Priority (highest → lowest)
1. CLI arguments (`--n_clusters 5`)
2. `pipeline_config.yaml` (local, git-ignored)
3. Environment variables (`ENSAGENT_API_KEY`, etc.)
4. Dataclass defaults in `PipelineConfig`

### Setup
```bash
cp pipeline_config.example.yaml pipeline_config.yaml
# Edit: set data_path, sample_id, and API credentials at minimum
```

### Recommended credential method (env vars)
```bash
export ENSAGENT_API_PROVIDER="openai"
export ENSAGENT_API_KEY="sk-..."
export ENSAGENT_API_MODEL="gpt-4o"
```

### Key config fields
| Field | Default | Notes |
|-------|---------|-------|
| `data_path` | required | Path to Visium data directory |
| `sample_id` | required | Sample identifier string |
| `n_clusters` | `7` | Number of spatial domains |
| `methods` | all 8 | Which clustering methods to run |
| `api_provider` | `""` | One of: azure, openai, anthropic, gemini, openrouter, … |
| `skip_tool_runner` | `false` | Skip Stage A |
| `skip_scoring` | `false` | Skip Stage B |
| `run_best` | `true` | Run Stage C |
| `run_annotation_multiagent` | `true` | Run Stage D |
| `vlm_off` | `false` | Disable visual scoring module |
| `conda_exe` | `mamba` | Package manager (`mamba` or `conda`) |

---

## Coding Conventions

- **Python 3.10+**, PEP 8, 4-space indentation.
- `snake_case` for files, functions, variables, and CLI flags; `PascalCase` for classes.
- Add type hints to new/edited public functions.
- No formatter/linter config committed; match the style of the surrounding module.
- Sub-module `requirements.txt` files are shims pointing to the root `requirements.txt` — do not add deps there directly.

---

## Testing

- Framework: `unittest` (stdlib only, no pytest).
- Location: `tests/test_*.py`.
- Tests must be deterministic, fast, and independent of external services.
- Use `unittest.mock.patch` to isolate LLM calls and subprocess execution.
- Some tests load modules via `importlib.util.spec_from_file_location` to avoid import side effects.
- No formal coverage threshold; exercise critical branches.

```bash
python -m unittest discover -s tests -v
```

---

## Commit & PR Guidelines

- Subject line: imperative mood, under 72 chars (e.g., `Add kNN smoothing option to BEST builder`).
- One logical change per commit.
- PRs must include: purpose, key files changed, commands used for validation, and screenshots for any UI changes.

---

## Security

- Never commit secrets. `pipeline_config.yaml` is git-ignored for this reason.
- Credentials via env vars are preferred over storing in YAML.
- `scoring/pic_analyze/.env` holds per-module Azure creds — keep it out of git.
- Large data artifacts (`.h5`, `.rds`, images) are git-ignored; do not force-add them.

---

## Useful File Locations

| What | Where |
|------|-------|
| Full project docs | `README.md` |
| Conda env setup guide (Chinese) | `ENV_SETUP.md` |
| Config template with all options | `pipeline_config.example.yaml` |
| AI agent dev guidelines | `AGENTS.md` |
| Tool registry (schemas + dispatch) | `ensagent_tools/registry.py` |
| Multi-provider LLM abstraction | `scoring/provider_runtime.py` |
| Scoring biological rubric | `scoring/gpt_scorer.py` (`RUBRIC_TEXT`) |
| Annotation agent schemas | `annotation/annotation_multiagent/schemas.py` |
| Web app launcher | `start.py` |
| Health check | `tools/health_check.py` |
| Chat API (SSE + tool-calling) | `api/routes/chat.py` |
| CLI chat agent | `ensagent_agent/chat.py` |
| Frontend state management | `frontend/lib/store.ts` |
| API config bridge (alias sync) | `api/deps.py` |
