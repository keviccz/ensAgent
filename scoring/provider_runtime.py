from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping


_PROVIDER_ALIASES = {
    "": "",
    "auto": "",
    "none": "",
    "together": "together_ai",
}

_DEFAULT_ENDPOINTS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together_ai": "https://api.together.xyz/v1",
    "mistral": "https://api.mistral.ai/v1",
    "cohere": "https://api.cohere.com/v1",
    "xai": "https://api.x.ai/v1",
    "perplexity": "https://api.perplexity.ai",
}

_PROVIDER_KEY_ENV: Dict[str, List[str]] = {
    "azure": ["AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "together_ai": ["TOGETHER_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "perplexity": ["PERPLEXITY_API_KEY"],
}


def normalize_provider(provider: str | None) -> str:
    normalized = str(provider or "").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def detect_provider(endpoint: str | None) -> str:
    endpoint_text = str(endpoint or "").strip().lower()
    if not endpoint_text:
        return ""
    if "openai.azure.com" in endpoint_text or "cognitiveservices.azure.com" in endpoint_text:
        return "azure"
    if "api.openai.com" in endpoint_text:
        return "openai"
    if "api.anthropic.com" in endpoint_text or "anthropic" in endpoint_text:
        return "anthropic"
    if "openrouter.ai" in endpoint_text:
        return "openrouter"
    if "api.deepseek.com" in endpoint_text:
        return "deepseek"
    if "api.groq.com" in endpoint_text:
        return "groq"
    if "api.together.xyz" in endpoint_text:
        return "together_ai"
    if "api.mistral.ai" in endpoint_text:
        return "mistral"
    if "api.cohere.com" in endpoint_text:
        return "cohere"
    if "api.x.ai" in endpoint_text:
        return "xai"
    if "api.perplexity.ai" in endpoint_text:
        return "perplexity"
    if "generativelanguage.googleapis.com" in endpoint_text:
        return "gemini"
    return ""


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


@dataclass
class ProviderConfig:
    provider: str
    api_key: str
    endpoint: str
    api_version: str
    model: str
    ocr_model: str


def resolve_provider_config(
    *,
    api_provider: str | None = None,
    api_key: str | None = None,
    api_endpoint: str | None = None,
    api_version: str | None = None,
    api_model: str | None = None,
    api_deployment: str | None = None,
    azure_openai_key: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
    azure_deployment: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ProviderConfig:
    env_map = dict(os.environ) if env is None else dict(env)

    endpoint_input = _first_non_empty(api_endpoint, azure_endpoint)
    provider = normalize_provider(api_provider) or detect_provider(endpoint_input) or "openai"

    key = _first_non_empty(
        api_key,
        azure_openai_key if provider == "azure" else "",
    )
    if not key:
        key_env_candidates = _PROVIDER_KEY_ENV.get(provider, [])
        for env_name in key_env_candidates:
            key = _first_non_empty(key, env_map.get(env_name, ""))
            if key:
                break
        key = _first_non_empty(key, env_map.get("ENSAGENT_API_KEY", ""))

    endpoint = _first_non_empty(
        api_endpoint,
        azure_endpoint if provider == "azure" else "",
        env_map.get("ENSAGENT_API_ENDPOINT", ""),
        _DEFAULT_ENDPOINTS.get(provider, ""),
    )
    version = _first_non_empty(
        api_version,
        azure_api_version if provider == "azure" else "",
        env_map.get("ENSAGENT_API_VERSION", ""),
        "2024-12-01-preview" if provider == "azure" else "",
    )
    model = _first_non_empty(
        api_model,
        api_deployment,
        azure_deployment if provider == "azure" else "",
        env_map.get("ENSAGENT_API_MODEL", ""),
        env_map.get("OPENAI_MODEL", ""),
        "gpt-4o",
    )
    ocr_model = _first_non_empty(
        env_map.get("ENSAGENT_OCR_MODEL", ""),
        model,
    )

    return ProviderConfig(
        provider=provider,
        api_key=key,
        endpoint=endpoint,
        api_version=version,
        model=model,
        ocr_model=ocr_model,
    )


def resolve_litellm_model(*, provider: str, model: str) -> str:
    cleaned_model = (model or "").strip() or "gpt-4o"
    if "/" in cleaned_model:
        return cleaned_model
    provider_prefix = {
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
    prefix = provider_prefix.get(provider)
    if prefix:
        return f"{prefix}/{cleaned_model}"
    return cleaned_model


def _extract_text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: List[str] = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return str(value)


def _extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices", [])
    if not choices:
        return ""
    choice0 = choices[0]
    message = getattr(choice0, "message", None)
    if message is None and isinstance(choice0, dict):
        message = choice0.get("message", {})
    if hasattr(message, "content"):
        return _extract_text_content(getattr(message, "content"))
    if isinstance(message, dict):
        return _extract_text_content(message.get("content"))
    return ""


def completion_raw(
    *,
    config: ProviderConfig,
    messages: List[Dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_tokens: int = 1200,
) -> Any:
    try:
        from litellm import completion  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "litellm is required for multi-provider runtime. Install with `pip install litellm`."
        ) from exc

    provider = normalize_provider(config.provider) or detect_provider(config.endpoint) or "openai"
    resolved_model = resolve_litellm_model(provider=provider, model=(model or config.model))

    kwargs: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.endpoint and provider in {
        "openai",
        "openai_compatible",
        "others",
        "azure",
        "openrouter",
        "deepseek",
        "groq",
        "together_ai",
        "mistral",
        "cohere",
        "xai",
        "perplexity",
    }:
        kwargs["api_base"] = config.endpoint
    if config.api_version and provider == "azure":
        kwargs["api_version"] = config.api_version
    return completion(**kwargs)


def completion_text(
    *,
    config: ProviderConfig,
    messages: List[Dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_tokens: int = 1200,
) -> str:
    response = completion_raw(
        config=config,
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return _extract_response_text(response)


def parse_json_text(raw_text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```json|```", "", str(raw_text or ""), flags=re.IGNORECASE).strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def completion_json(
    *,
    config: ProviderConfig,
    messages: List[Dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    text = completion_text(
        config=config,
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return parse_json_text(text)

