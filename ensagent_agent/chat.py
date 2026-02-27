"""
EnsAgent CLI chat agent with LLM tool-calling.

All tool definitions and implementations are imported from ``ensagent_tools``.
This file is a thin REPL that wires the LLM to the tool registry.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from litellm import completion

from ensagent_tools import TOOL_SCHEMAS, execute_tool, load_config, PipelineConfig


def _detect_provider(endpoint: str) -> str | None:
    endpoint_lower = (endpoint or "").strip().lower()
    if not endpoint_lower:
        return None
    if "openai.azure.com" in endpoint_lower or "cognitiveservices.azure.com" in endpoint_lower:
        return "azure"
    if "api.openai.com" in endpoint_lower:
        return "openai"
    if "api.anthropic.com" in endpoint_lower or "anthropic" in endpoint_lower:
        return "anthropic"
    if "openrouter.ai" in endpoint_lower:
        return "openrouter"
    if "api.deepseek.com" in endpoint_lower:
        return "deepseek"
    if "api.groq.com" in endpoint_lower:
        return "groq"
    if "api.together.xyz" in endpoint_lower:
        return "together_ai"
    if "api.mistral.ai" in endpoint_lower:
        return "mistral"
    if "api.cohere.com" in endpoint_lower:
        return "cohere"
    if "api.x.ai" in endpoint_lower:
        return "xai"
    if "api.perplexity.ai" in endpoint_lower:
        return "perplexity"
    if "generativelanguage.googleapis.com" in endpoint_lower:
        return "gemini"
    return None


def _normalize_provider(provider: str | None) -> str:
    normalized = (provider or "").strip().lower()
    if normalized in {"", "auto", "none"}:
        return ""
    aliases = {
        "together": "together_ai",
    }
    return aliases.get(normalized, normalized)


def _resolve_litellm_model(provider: str, model: str) -> str:
    cleaned = (model or "").strip() or "gpt-4o"
    if "/" in cleaned:
        return cleaned
    prefix_map = {
        "azure": "azure",
        "anthropic": "anthropic",
        "gemini": "gemini",
        "openrouter": "openrouter",
        "deepseek": "deepseek",
        "groq": "groq",
        "together_ai": "together_ai",
        "mistral": "mistral",
        "cohere": "cohere",
        "xai": "xai",
        "perplexity": "perplexity",
    }
    prefix = prefix_map.get(provider)
    return f"{prefix}/{cleaned}" if prefix else cleaned


def _resolve_llm_config(cfg: PipelineConfig) -> Dict[str, str]:
    provider = _normalize_provider(cfg.api_provider) or _detect_provider(cfg.api_endpoint) or ""
    api_key = cfg.api_key or cfg.azure_openai_key or os.environ.get("AZURE_OPENAI_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    endpoint = cfg.api_endpoint or cfg.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "") or os.environ.get("OPENAI_API_BASE", "")
    api_version = cfg.api_version or cfg.azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    model = (
        cfg.api_model
        or cfg.api_deployment
        or cfg.azure_deployment
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        or os.environ.get("OPENAI_MODEL", "")
        or "gpt-4o"
    )

    if not provider:
        provider = _detect_provider(endpoint) or "openai"

    if not api_key:
        raise RuntimeError("API key missing. Set pipeline_config.yaml api_key or env vars.")
    if provider in {"azure", "openai_compatible", "others"} and not endpoint:
        raise RuntimeError(f"API endpoint missing for provider '{provider}'.")

    return {
        "provider": provider,
        "api_key": api_key,
        "endpoint": endpoint,
        "api_version": api_version,
        "model": model,
    }


_SYSTEM_PROMPT = """\
You are EnsAgent, a specialist assistant for spatial transcriptomics ensemble analysis.

## What EnsAgent Does
EnsAgent integrates eight spatial clustering methods (IRIS, BASS, DR-SC, BayesSpace, \
SEDR, GraphST, STAGATE, stLearn) into a four-stage pipeline that produces consensus \
spatial domain labels from 10X Visium data.

## Pipeline Stages
  A. Tool-Runner  — Execute clustering methods in R/PY/PY2 environments, align labels, \
produce per-method spot/DEG/pathway CSVs.
  B. Scoring      — LLM-driven evaluation of each method per domain, build scores_matrix \
and labels_matrix.
  C. BEST Builder — Select optimal domain label for every spot from the score/label \
matrices, output BEST_<sample>_{spot,DEGs,PATHWAY}.csv + result image.
  D. Annotation   — Multi-agent (VLM + Peer + Critic) annotation of spatial domains.

Stages depend on each other: A → B → C → D.  Partial runs are supported \
(e.g. skip_tool_runner when Stage A outputs already exist).

## Available Tools
| Tool              | Purpose |
|-------------------|---------|
| check_envs        | Verify R / PY / PY2 mamba environments exist |
| setup_envs        | Create missing environments from envs/*.yml |
| run_tool_runner   | Execute Stage A (requires data_path + sample_id) |
| run_scoring       | Execute Stage B |
| run_best_builder  | Execute Stage C (requires sample_id) |
| run_annotation    | Execute Stage D (requires sample_id) |
| run_end_to_end    | Run the full A→B→C→D pipeline |
| show_config       | Display current pipeline_config.yaml |
| set_config        | Update a single key in pipeline_config.yaml |

## Policies
1. Before running any stage, call show_config to confirm data_path and sample_id are set. \
If empty, ask the user or call set_config.
2. Before the first run, call check_envs. If environments are missing, call setup_envs.
3. When the user asks to "run everything", use run_end_to_end after confirming config.
4. Explain briefly what each stage will do before launching it.
5. When a tool returns {"ok": false}, report the error clearly and suggest a fix.
6. Respond in the same language the user writes in.
7. Be concise. Prefer structured lists over long paragraphs.
"""


def main() -> None:
    cfg = load_config()
    llm_cfg = _resolve_llm_config(cfg)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    print("ensAgent LLM Agent (tool-calling)")
    print("Examples:")
    print("- Please set up the environments.")
    print("- Please run end-to-end.")
    print("- Show config.")
    print("- Set data_path to D:\\DATA\\... ; Set sample_id to DLPFC_151507")
    print("Type: exit\n")

    while True:
        try:
            user = input("ensAgent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if user.lower() in {"exit", "quit"}:
            print("bye")
            return

        messages.append({"role": "user", "content": user})

        for _ in range(8):
            kwargs: Dict[str, Any] = {
                "model": _resolve_litellm_model(llm_cfg["provider"], llm_cfg["model"]),
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "tool_choice": "auto",
                "temperature": 0.2,
                "api_key": llm_cfg["api_key"],
            }
            if llm_cfg["endpoint"] and llm_cfg["provider"] in {
                "azure",
                "openai",
                "openai_compatible",
                "others",
                "openrouter",
                "deepseek",
                "groq",
                "together_ai",
                "mistral",
                "cohere",
                "xai",
                "perplexity",
            }:
                kwargs["api_base"] = llm_cfg["endpoint"]
            if llm_cfg["provider"] == "azure":
                kwargs["api_version"] = llm_cfg["api_version"]

            resp = completion(**kwargs)
            payload = resp.model_dump() if hasattr(resp, "model_dump") else resp
            msg = (payload.get("choices") or [{}])[0].get("message", {}) or {}

            content = msg.get("content") or ""
            if content:
                print(str(content).strip())
                messages.append({"role": "assistant", "content": str(content)})

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                break

            for tc in tool_calls:
                function = tc.get("function", {}) or {}
                name = function.get("name", "")
                raw_args = function.get("arguments") or "{}"
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                out = execute_tool(name, args, cfg)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": json.dumps(out, ensure_ascii=False),
                })

            # Reload config after set_config calls
            cfg = load_config()


if __name__ == "__main__":
    main()
