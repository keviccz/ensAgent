# Chat Tool-Call + UI Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the silent chat (missing `completion_text_stream`), implement function-calling loop in the API chat route, fix three UI bugs, repair the scores data API, clean up root-level clutter, and run a scoring smoke-test.

**Architecture:** The FastAPI `api/routes/chat.py` route becomes a tool-calling agent loop: it calls the LLM with `TOOL_SCHEMAS`, executes any returned tool calls via `execute_tool`, then streams the final text back to the frontend as SSE. The frontend already renders `ToolCallBlock` components — it just needs to receive the right `{type:"tool_call"}` SSE events. No new files are needed; all changes are in existing files.

**Tech Stack:** FastAPI + litellm streaming, `ensagent_tools.TOOL_SCHEMAS / execute_tool`, asyncio.to_thread, Next.js / Zustand (frontend state), recharts (ScoresMatrix)

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `api/routes/chat.py` | Modify | Fix `resolve_provider_config(**cfg)`, implement litellm streaming + tool-call loop |
| `scoring/provider_runtime.py` | Modify | Add `completion_text_stream` async generator |
| `frontend/components/analysis/ScoresMatrix.tsx` | Modify | Remove hard-coded `D` prefix on column headers |
| `frontend/components/chat/ChatMessages.tsx` | Modify | Raise empty-state opacity from 0.5 → 0.9 |
| `frontend/lib/types.ts` | Modify | Add `ToolCallEvent` SSE type used by chat stream |
| `frontend/lib/api.ts` | Modify | Parse new `tool_call` SSE events alongside `delta` |
| `frontend/app/chat/page.tsx` | Modify | Handle `tool_call` events, call `addToolCall` on active message |
| `frontend/lib/store.ts` | Modify | Add `addToolCall` action; store tool calls on streaming message |
| `api/routes/data.py` | Modify | Fix scores endpoint: aggregate spot-level CSV into method×domain matrix |
| Root directory | Cleanup | Move design/debug PNGs to `docs/archive/design/`, delete `tmp*/` dirs |

---

## Task 1: Fix `completion_text_stream` + chat route

**Root cause:** `api/routes/chat.py` calls `completion_text_stream` (doesn't exist in `provider_runtime.py`) and passes `cfg` dict as positional arg to `resolve_provider_config` (which requires keyword args). This causes an immediate `ImportError` / `TypeError` so no reply is ever sent.

**Files:**
- Modify: `scoring/provider_runtime.py` (add async generator after `completion_text`)
- Modify: `api/routes/chat.py` (fix call signature + use new function)

- [ ] **Step 1: Add `completion_text_stream` to `scoring/provider_runtime.py`**

Add this function after the existing `completion_text` function (around line 383):

```python
async def completion_text_stream(
    config: "ProviderConfig",
    messages: List[Dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 2048,
):
    """Async generator that yields text delta chunks from the LLM."""
    import asyncio

    provider = normalize_provider(config.provider) or detect_provider(config.endpoint) or "openai"
    requested_model = (model or config.model or "gpt-4o")
    resolved_model = resolve_litellm_model(provider=provider, model=requested_model)

    kwargs: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": True,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.endpoint and provider in {
        "openai", "openai_compatible", "others", "azure",
        "openrouter", "deepseek", "groq", "together_ai",
        "mistral", "cohere", "xai", "perplexity",
    }:
        kwargs["api_base"] = config.endpoint
    if config.api_version and provider == "azure":
        kwargs["api_version"] = config.api_version

    def _sync_stream():
        from litellm import completion as litellm_completion  # type: ignore
        return litellm_completion(**kwargs)

    loop = asyncio.get_event_loop()
    stream = await loop.run_in_executor(None, _sync_stream)

    def _iter_stream():
        for chunk in stream:
            delta = ""
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                pass
            if delta:
                yield delta

    for delta in await loop.run_in_executor(None, lambda: list(_iter_stream())):
        yield delta
```

- [ ] **Step 2: Rewrite `api/routes/chat.py` with correct kwarg unpacking + system prompt**

```python
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from api.deps import load_config

router = APIRouter(prefix="/api")

SYSTEM_PROMPT = (
    "You are EnsAgent, an AI assistant for spatial transcriptomics analysis. "
    "You help researchers run the EnsAgent pipeline (Tool-Runner → Scoring → BEST → Annotation), "
    "interpret clustering results, and annotate spatial domains in 10x Visium data. "
    "Use the available tools to run pipeline stages when the user asks. "
    "Be concise and scientific."
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

async def _stream_response(message: str, history: list, cfg: dict):
    """Yield SSE chunks. Handles plain text; tool-calling is in Task 3."""
    from scoring.provider_runtime import resolve_provider_config, completion_text_stream

    provider_cfg = resolve_provider_config(**{k: v for k, v in cfg.items() if isinstance(v, str)})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    async for chunk in completion_text_stream(provider_cfg, messages):
        yield f"data: {json.dumps({'delta': chunk})}\n\n"
    yield "data: [DONE]\n\n"

@router.post("/chat")
async def chat(req: ChatRequest):
    cfg = load_config()
    return StreamingResponse(
        _stream_response(req.message, [m.dict() for m in req.history], cfg),
        media_type="text/event-stream",
    )
```

- [ ] **Step 3: Verify chat works end-to-end**

With FastAPI running (`conda run -n ens_dev python -m uvicorn api.main:app --port 8000 --reload`):

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello, what can you do?","history":[]}' 
```

Expected: SSE stream of `data: {"delta": "..."}` lines followed by `data: [DONE]`.

- [ ] **Step 4: Commit**

```bash
git add scoring/provider_runtime.py api/routes/chat.py
git commit -m "fix(chat): add completion_text_stream and fix resolve_provider_config call"
```

---

## Task 2: UI Quick Fixes

**Files:**
- Modify: `frontend/components/analysis/ScoresMatrix.tsx:69`
- Modify: `frontend/components/chat/ChatMessages.tsx:32`

### 2a: Remove hard-coded `D` prefix from ScoresMatrix column headers

The CSV has method names (BASS, BayesSpace…) as column keys. `D{d}` turns "BASS" → "DBASS". Fix: only add "D" when `d` is a digit string (a domain number).

- [ ] **Step 1: Fix ScoresMatrix.tsx line 69**

Change:
```tsx
                  D{d}
```
To:
```tsx
                  {isNaN(Number(d)) ? d : `D${d}`}
```

- [ ] **Step 2: Verify in browser — Analysis page columns now show "BASS", "BayesSpace", etc.**

Navigate to `http://localhost:3000/analysis` and confirm column headers no longer have a leading "D".

### 2b: Fix empty-state fade in ChatMessages

The placeholder ("Start a conversation with EnsAgent") container has `opacity: 0.5` which makes it very faint.

- [ ] **Step 3: Fix ChatMessages.tsx line 32 — raise opacity**

Change `opacity: 0.5` to `opacity: 0.85`.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/analysis/ScoresMatrix.tsx frontend/components/chat/ChatMessages.tsx
git commit -m "fix(ui): remove D-prefix from scores columns, fix empty-state opacity"
```

---

## Task 3: Tool-Calling Chat Agent Loop

**Goal:** When a user asks EnsAgent to run a pipeline stage (e.g., "run scoring"), the LLM should emit a tool call, the backend should execute it via `execute_tool`, and the result should be streamed back with `{type:"tool_call"}` events so the frontend can render `ToolCallBlock`.

**Files:**
- Modify: `api/routes/chat.py` (replace `_stream_response` with tool-calling loop)
- Modify: `frontend/lib/types.ts` (add `StreamEvent` discriminated union)
- Modify: `frontend/lib/api.ts` (parse `tool_call` events)
- Modify: `frontend/lib/store.ts` (add `addToolCall` + `StreamingToolCall` state)
- Modify: `frontend/app/chat/page.tsx` (dispatch `tool_call` events to store)
- Modify: `frontend/components/chat/ChatMessages.tsx` (show streaming tool calls)

### 3a: Backend — tool-calling loop in `api/routes/chat.py`

- [ ] **Step 1: Rewrite `api/routes/chat.py` with full tool-calling loop**

```python
import json
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Any, Dict
from api.deps import load_config

router = APIRouter(prefix="/api")

SYSTEM_PROMPT = (
    "You are EnsAgent, an AI assistant for spatial transcriptomics analysis. "
    "You help researchers run the EnsAgent pipeline (Tool-Runner → Scoring → BEST → Annotation), "
    "interpret clustering results, and annotate spatial domains in 10x Visium data. "
    "Use the available tools when asked to run a pipeline stage. "
    "Be concise and scientific."
)

MAX_TOOL_ROUNDS = 5  # prevent infinite loops

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


def _sync_completion_with_tools(messages: list, cfg: dict, stream: bool = False) -> Any:
    """Call litellm with TOOL_SCHEMAS. Returns raw response or stream."""
    from scoring.provider_runtime import resolve_provider_config, resolve_litellm_model, normalize_provider, detect_provider
    from ensagent_tools import TOOL_SCHEMAS
    import litellm  # type: ignore

    str_cfg = {k: v for k, v in cfg.items() if isinstance(v, (str, type(None)))}
    provider_cfg = resolve_provider_config(**str_cfg)
    provider = normalize_provider(provider_cfg.provider) or detect_provider(provider_cfg.endpoint) or "openai"
    resolved_model = resolve_litellm_model(provider=provider, model=provider_cfg.model or "gpt-4o")

    kwargs: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "tool_choice": "auto",
        "temperature": float(cfg.get("temperature", 0.7)),
        "top_p": float(cfg.get("top_p", 1.0)),
        "max_tokens": 2048,
        "stream": stream,
    }
    if provider_cfg.api_key:
        kwargs["api_key"] = provider_cfg.api_key
    if provider_cfg.endpoint and provider in {
        "openai", "openai_compatible", "others", "azure",
        "openrouter", "deepseek", "groq", "together_ai",
        "mistral", "cohere", "xai", "perplexity",
    }:
        kwargs["api_base"] = provider_cfg.endpoint
    if provider_cfg.api_version and provider == "azure":
        kwargs["api_version"] = provider_cfg.api_version

    return litellm.completion(**kwargs)


async def _agent_stream(message: str, history: list, cfg: dict):
    """Tool-calling agent loop: yields SSE events."""
    from ensagent_tools import execute_tool
    from ensagent_tools.config_manager import PipelineConfig

    # Build message list
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    loop = asyncio.get_event_loop()

    for _round in range(MAX_TOOL_ROUNDS):
        # Non-streaming call to check for tool calls first
        response = await loop.run_in_executor(
            None, lambda: _sync_completion_with_tools(messages, cfg, stream=False)
        )

        choice = response.choices[0]
        msg = choice.message

        # --- Tool call branch ---
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            # Emit each tool call as an SSE event
            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args_str = tc.function.arguments or "{}"
                try:
                    fn_args = json.loads(fn_args_str)
                except json.JSONDecodeError:
                    fn_args = {}

                yield f"data: {json.dumps({'type': 'tool_call', 'name': fn_name, 'args': fn_args, 'id': tc.id})}\n\n"

                # Execute tool
                pipe_cfg = PipelineConfig(**{
                    k: v for k, v in cfg.items()
                    if k in PipelineConfig.__dataclass_fields__
                }) if hasattr(PipelineConfig, "__dataclass_fields__") else cfg

                result = await loop.run_in_executor(
                    None, lambda: execute_tool(fn_name, fn_args, pipe_cfg)
                )
                result_text = json.dumps(result)
                yield f"data: {json.dumps({'type': 'tool_result', 'name': fn_name, 'result': result_text, 'id': tc.id})}\n\n"

                # Append to messages for next round
                messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": fn_name, "arguments": fn_args_str}}
                ]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
            continue  # next round → LLM responds to tool results

        # --- Text response branch (streaming) ---
        response_stream = await loop.run_in_executor(
            None, lambda: _sync_completion_with_tools(messages, cfg, stream=True)
        )

        def _iter():
            for chunk in response_stream:
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if delta:
                    yield delta

        for delta in await loop.run_in_executor(None, lambda: list(_iter())):
            yield f"data: {json.dumps({'type': 'delta', 'delta': delta})}\n\n"
        break  # done

    yield "data: [DONE]\n\n"


@router.post("/chat")
async def chat(req: ChatRequest):
    cfg = load_config()
    return StreamingResponse(
        _agent_stream(req.message, [m.dict() for m in req.history], cfg),
        media_type="text/event-stream",
    )
```

- [ ] **Step 2: Commit backend**

```bash
git add api/routes/chat.py
git commit -m "feat(api): implement tool-calling agent loop in chat route"
```

### 3b: Frontend — parse and display tool call events

- [ ] **Step 3: Update `frontend/lib/types.ts` — add StreamEvent union**

Add after the `ChatMessage` type:

```ts
export type StreamEvent =
  | { type: 'delta'; delta: string }
  | { type: 'tool_call'; name: string; args: Record<string, unknown>; id: string }
  | { type: 'tool_result'; name: string; result: string; id: string }
```

Also update `ToolCall` if it's used in `ChatMessage`:

```ts
export interface ToolCall {
  id: string
  name: string
  args: Record<string, unknown>
  result?: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  timestamp: number
  toolCalls?: ToolCall[]
  reasoning?: ReasoningStep[]
}
```

- [ ] **Step 4: Update `frontend/lib/store.ts` — add streaming tool calls + `addToolCall` action**

Add to state interface and initial state:

```ts
// in AppState:
streamingToolCalls: ToolCall[]

// in initial state:
streamingToolCalls: [],
```

Add action:

```ts
addStreamingToolCall: (tc: ToolCall) =>
  set((s) => ({ streamingToolCalls: [...s.streamingToolCalls, tc] })),

updateStreamingToolCall: (id: string, result: string) =>
  set((s) => ({
    streamingToolCalls: s.streamingToolCalls.map((tc) =>
      tc.id === id ? { ...tc, result } : tc
    ),
  })),
```

Update `finalizeStream` to attach `streamingToolCalls` to the message and clear them:

```ts
finalizeStream: (conversationId: string) =>
  set((s) => {
    const msg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      text: s.streamingMessage,
      timestamp: Date.now(),
      toolCalls: s.streamingToolCalls.length > 0 ? [...s.streamingToolCalls] : undefined,
    }
    return {
      conversations: s.conversations.map((c) =>
        c.id === conversationId ? { ...c, messages: [...c.messages, msg] } : c
      ),
      streamingMessage: '',
      streamingToolCalls: [],
    }
  }),
```

- [ ] **Step 5: Update `frontend/lib/api.ts` — parse all SSE event types**

Replace `createChatStream` with:

```ts
export function createChatStream(
  message: string,
  history: { role: string; content: string }[],
  onChunk: (chunk: string) => void,
  onToolCall: (tc: { id: string; name: string; args: Record<string, unknown> }) => void,
  onToolResult: (id: string, result: string) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): () => void {
  const controller = new AbortController()

  fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(`Chat error ${res.status}`)
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const text = decoder.decode(value)
        for (const line of text.split('\n')) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (raw === '[DONE]') { onDone(); return }
          try {
            const ev = JSON.parse(raw) as { type?: string; delta?: string; name?: string; args?: Record<string, unknown>; id?: string; result?: string }
            if (!ev.type || ev.type === 'delta') {
              onChunk(ev.delta ?? '')
            } else if (ev.type === 'tool_call') {
              onToolCall({ id: ev.id!, name: ev.name!, args: ev.args ?? {} })
            } else if (ev.type === 'tool_result') {
              onToolResult(ev.id!, ev.result ?? '')
            }
          } catch { /* skip */ }
        }
      }
      onDone()
    })
    .catch((err: Error) => { if (err.name !== 'AbortError') onError(err) })

  return () => controller.abort()
}
```

- [ ] **Step 6: Update `frontend/app/chat/page.tsx` — dispatch tool call events**

Replace `createChatStream` call:

```tsx
const { addStreamingToolCall, updateStreamingToolCall } = useStore()

createChatStream(
  text,
  history,
  (chunk) => appendStreamChunk(chunk),
  (tc) => addStreamingToolCall(tc),
  (id, result) => updateStreamingToolCall(id, result),
  () => finalizeStream(activeConversationId!),
  (err) => {
    finalizeStream(activeConversationId!)
    addMessage(activeConversationId!, {
      id: crypto.randomUUID(),
      role: 'assistant',
      text: `⚠ Connection error: ${err.message}\n\nMake sure the FastAPI server is running on localhost:8000 and your API credentials are configured in Settings.`,
      timestamp: Date.now(),
    })
  },
)
```

Also add streaming tool calls display in the JSX above `<ChatMessages>`:

```tsx
// In ChatPage JSX, pass streamingToolCalls to ChatMessages:
const { streamingToolCalls } = useStore()

<ChatMessages
  messages={messages}
  streamingText={isStreaming ? streamingMessage : undefined}
  streamingToolCalls={isStreaming ? streamingToolCalls : []}
/>
```

- [ ] **Step 7: Update `frontend/components/chat/ChatMessages.tsx` — show streaming tool calls**

Add `streamingToolCalls?: ToolCall[]` to `Props` and render them before the streaming text cursor:

```tsx
import { ToolCallBlock } from './ToolCallBlock'
import type { ToolCall } from '@/lib/types'

interface Props {
  messages: ChatMessage[]
  streamingText?: string
  streamingToolCalls?: ToolCall[]
}

// In JSX, before the streamingText block:
{streamingToolCalls && streamingToolCalls.length > 0 && (
  <div style={{ marginBottom: '8px' }}>
    {streamingToolCalls.map((tc) => (
      <ToolCallBlock
        key={tc.id}
        name={tc.name}
        args={tc.args}
        result={tc.result}
      />
    ))}
  </div>
)}
```

- [ ] **Step 8: Verify ToolCallBlock accepts the right props (check existing component)**

Read `frontend/components/chat/ToolCallBlock.tsx` and confirm its props match `{name, args, result}`. Adjust if needed.

- [ ] **Step 9: Run `npx tsc --noEmit` in `frontend/`**

Expected: zero errors.

- [ ] **Step 10: Commit frontend**

```bash
git add frontend/
git commit -m "feat(frontend): parse and render tool_call SSE events in chat"
```

---

## Task 4: Fix Scores Data API

**Problem:** `scores_matrix.csv` rows = spot IDs, columns = methods. The frontend expects rows = methods, columns = domain IDs. Need to join with `labels_matrix.csv` to average scores per domain per method.

**Files:**
- Modify: `api/routes/data.py` (rewrite `get_scores`)

- [ ] **Step 1: Rewrite `get_scores` in `api/routes/data.py`**

```python
@router.get("/scores")
def get_scores(sample_id: str = Query("DLPFC_151507")):
    scores_path = Path("scoring/output/consensus/scores_matrix.csv")
    labels_path = Path("scoring/output/consensus/labels_matrix.csv")
    if not scores_path.exists() or not labels_path.exists():
        return {"rows": []}

    # Load scores: spot_id → {method: score}
    scores: dict = {}
    methods: list = []
    with open(scores_path, newline="") as f:
        reader = csv.DictReader(f)
        methods = [c for c in (reader.fieldnames or []) if c]  # skip empty first column
        for row in reader:
            spot_id = row.get("", row.get("spot_id", ""))
            scores[spot_id] = {m: float(row.get(m, 0) or 0) for m in methods}

    # Load labels: spot_id → {method: domain_label}
    labels: dict = {}
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spot_id = row.get("", row.get("spot_id", ""))
            labels[spot_id] = {m: str(row.get(m, "?")) for m in methods}

    # Build method × domain aggregation
    from collections import defaultdict
    method_domain_scores: dict = {m: defaultdict(list) for m in methods}
    for spot_id, spot_scores in scores.items():
        spot_labels = labels.get(spot_id, {})
        for m in methods:
            domain = spot_labels.get(m, "?")
            method_domain_scores[m][domain].append(spot_scores.get(m, 0.0))

    rows = []
    for m in methods:
        domain_avgs = {
            domain: round(sum(vals) / len(vals), 4)
            for domain, vals in sorted(method_domain_scores[m].items())
            if vals
        }
        rows.append({"method": m, "scores": domain_avgs})

    return {"rows": rows}
```

- [ ] **Step 2: Verify via curl**

```bash
curl -s "http://localhost:8000/api/data/scores?sample_id=DLPFC_151507" | python -m json.tool | head -30
```

Expected: rows with `method` = "BASS", "BayesSpace", etc. and `scores` keyed by domain labels ("1","2",…,"7").

- [ ] **Step 3: Commit**

```bash
git add api/routes/data.py
git commit -m "fix(api): aggregate scores_matrix by domain label for per-method display"
```

---

## Task 5: Directory Cleanup

**Files:**
- Move: All `design_*.png`, `debug_*.png`, `pages_*.png`, `v2_*.png`, `v3_*.png` → `docs/archive/design/`
- Delete: `tmp*/` directories in project root
- (Do NOT touch `ARI&Picture_DLPFC_151507/` — may contain research data)

- [ ] **Step 1: Create docs/archive/design/ and move screenshots**

```bash
mkdir -p "docs/archive/design"
mv design_*.png debug_*.png pages_*.png v2_*.png v3_*.png docs/archive/design/ 2>/dev/null || true
```

- [ ] **Step 2: Remove tmp directories**

```bash
rm -rf tmp*/  2>/dev/null || true
```

- [ ] **Step 3: Verify root directory is clean**

```bash
ls *.png 2>/dev/null || echo "No loose PNGs in root"
ls -d tmp*/ 2>/dev/null || echo "No tmp dirs"
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: move design screenshots to docs/archive/design, remove tmp dirs"
```

---

## Task 6: Smoke-Test Scoring Pipeline

**Goal:** Confirm `run_scoring` executes successfully end-to-end with existing `scoring/input/` data.

- [ ] **Step 1: Run scoring via execute_tool**

```bash
conda run -n ens_dev python -c "
import sys
sys.path.insert(0, '.')
from ensagent_tools import execute_tool
from ensagent_tools.config_manager import load_config
cfg = load_config()
print('Config loaded:', cfg.sample_id, cfg.api_provider)
result = execute_tool('run_scoring', {}, cfg)
print('Result:', result)
"
```

Expected: `{'ok': True, ...}` with output written to `scoring/output/consensus/`.

- [ ] **Step 2: Verify output files updated**

```bash
ls -la scoring/output/consensus/
head -2 scoring/output/consensus/scores_matrix.csv
```

Expected: `scores_matrix.csv` and `labels_matrix.csv` present with fresh timestamps.

- [ ] **Step 3: Reload Analysis page in browser and verify ScoresMatrix shows data**

Navigate to `http://localhost:3000/analysis`. The Scores Matrix table should now show 8 method rows with averaged domain scores in each cell.

- [ ] **Step 4: Commit any fixes found during smoke test**

```bash
git add -A
git commit -m "test: verify scoring pipeline smoke test passes"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Chat silent → fixed via `completion_text_stream` + correct kwarg unpacking (Task 1)
- [x] Empty-state opacity too low → fixed (Task 2b)
- [x] ScoresMatrix "DBASS" → fixed (Task 2a)
- [x] Tool-calling added to chat agent (Task 3)
- [x] Frontend renders tool call events (Task 3b)
- [x] Scores data API aggregated correctly (Task 4)
- [x] Directory cleaned up (Task 5)
- [x] Scoring pipeline smoke test (Task 6)

**Type consistency:** `ToolCall.args` is `Record<string, unknown>` in types.ts and the SSE parser, consistent throughout.

**Placeholder scan:** All code steps are complete. No TBDs.
