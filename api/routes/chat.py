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
    "You help researchers run the EnsAgent pipeline (Tool-Runner → Scoring → BEST → Annotation) "
    "and interpret spatial domain results for 10x Visium data.\n\n"
    "VISUALIZATION TOOLS (each renders directly in chat — call ONCE per request, then respond with text):\n"
    "- get_cluster_image: static PNG cluster map.\n"
    "- show_cluster_scatter: interactive clickable scatter plot (preferred for exploration).\n"
    "- show_scores_matrix: method × domain score heatmap table.\n"
    "- show_distributions: Domain Distribution bar chart + Method Ranking bar chart.\n\n"
    "IMPORTANT: After calling any visualization tool and receiving 'chart_rendered_in_chat: true' "
    "or 'image_rendered_in_chat: true', do NOT call more tools. Respond immediately with a brief "
    "text description of what was shown.\n\n"
    "OTHER TOOLS:\n"
    "- get_annotation_result: read existing domain annotations instantly (no pipeline run).\n"
    "- run_tool_runner / run_scoring / run_best_builder / run_annotation: "
    "execute pipeline stages — only when user explicitly asks to run a stage (takes minutes–hours).\n"
    "- python_use: run ad-hoc Python code for data queries.\n\n"
    "Be concise and scientific in your responses."
)

MAX_TOOL_ROUNDS = 5

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


def _sync_completion(messages: list, cfg: dict, stream: bool = False) -> Any:
    """Call litellm with TOOL_SCHEMAS. Returns raw response or stream."""
    from scoring.provider_runtime import resolve_provider_config, resolve_litellm_model, normalize_provider, detect_provider
    from ensagent_tools import TOOL_SCHEMAS
    import litellm  # type: ignore

    _API_KEYS = {
        "api_provider", "api_key", "api_endpoint", "api_version", "api_model",
        "azure_openai_key", "azure_endpoint", "azure_api_version", "azure_deployment",
    }
    str_cfg = {k: str(v) for k, v in cfg.items() if k in _API_KEYS and v is not None}
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


def _args_to_python(name: str, args: dict) -> str:
    """Generate a Python snippet representation of the tool call."""
    kw = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
    return f"execute_tool({repr(name)}, {{{kw}}})"


async def _agent_stream(message: str, history: list, cfg: dict):
    """Tool-calling agent loop: yields SSE events."""
    from ensagent_tools.registry import execute_tool, TOOL_SCHEMAS
    from ensagent_tools.config_manager import PipelineConfig, load_config as load_pipe_cfg

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    loop = asyncio.get_event_loop()

    # Load PipelineConfig for execute_tool
    pipe_cfg = load_pipe_cfg()

    for _round in range(MAX_TOOL_ROUNDS):
        # Non-streaming first call to detect tool calls — 90s timeout guards against LLM hang
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: _sync_completion(messages, cfg, stream=False)),
                timeout=90.0,
            )
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'delta', 'delta': 'LLM response timed out (>90s). Check your API connection in Settings.'})}\n\n"
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'type': 'delta', 'delta': f'Error: {e}'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if tool_calls:
            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args_str = tc.function.arguments or "{}"
                try:
                    fn_args = json.loads(fn_args_str)
                except json.JSONDecodeError:
                    fn_args = {}

                # Build ToolCall event for frontend
                tool_call_event = {
                    "type": "tool_call",
                    "id": tc.id,
                    "toolName": fn_name,
                    "args": [{"key": k, "value": str(v)} for k, v in fn_args.items()],
                    "jsonPayload": json.dumps(fn_args, indent=2),
                    "pythonSnippet": _args_to_python(fn_name, fn_args),
                }
                yield f"data: {json.dumps(tool_call_event)}\n\n"

                # Execute tool — 3 minute timeout to prevent hanging
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda: execute_tool(fn_name, fn_args, pipe_cfg)
                        ),
                        timeout=180.0,
                    )
                except asyncio.TimeoutError:
                    result = {
                        "ok": False,
                        "error": (
                            f"Tool '{fn_name}' timed out after 3 minutes. "
                            "For long-running pipeline stages, use the Pipeline controls in the UI."
                        ),
                    }
                result_text = json.dumps(result)

                yield f"data: {json.dumps({'type': 'tool_result', 'id': tc.id, 'result': result_text})}\n\n"

                # Strip large/non-serializable fields before feeding result back to the LLM.
                # image_b64: 500 KB base64 PNG would exceed token limits.
                # _chart: frontend-only token; LLM doesn't need the raw data.
                if isinstance(result, dict) and ("image_b64" in result or "_chart" in result):
                    result_for_llm = {k: v for k, v in result.items()
                                      if k not in ("image_b64", "_chart")}
                    if "image_b64" in result:
                        result_for_llm["image_rendered_in_chat"] = True
                    if "_chart" in result:
                        result_for_llm["chart_rendered_in_chat"] = True
                    result_text_for_llm = json.dumps(result_for_llm)
                else:
                    result_text_for_llm = result_text

                # Append assistant tool call + tool result to messages
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": fn_name, "arguments": fn_args_str},
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text_for_llm,
                })
            continue  # next round: LLM responds to tool results

        # Text response — stream it
        try:
            stream_resp = await loop.run_in_executor(
                None, lambda: _sync_completion(messages, cfg, stream=True)
            )

            def _collect():
                deltas = []
                for chunk in stream_resp:
                    try:
                        d = chunk.choices[0].delta.content or ""
                    except Exception:
                        d = ""
                    if d:
                        deltas.append(d)
                return deltas

            deltas = await loop.run_in_executor(None, _collect)
            for d in deltas:
                yield f"data: {json.dumps({'type': 'delta', 'delta': d})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'delta', 'delta': f'Error: {e}'})}\n\n"
        break

    yield "data: [DONE]\n\n"


@router.post("/chat")
async def chat(req: ChatRequest):
    cfg = load_config()
    return StreamingResponse(
        _agent_stream(req.message, [m.dict() for m in req.history], cfg),
        media_type="text/event-stream",
    )
