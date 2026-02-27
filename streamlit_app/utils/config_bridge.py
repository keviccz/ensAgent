"""
Helpers that bridge Streamlit settings and pipeline_config.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PIPELINE_CONFIG_NAME = "pipeline_config.yaml"


def _normalize_provider(provider: str | None) -> str:
    if provider is None:
        return ""
    normalized = str(provider).strip().lower()
    if normalized in {"", "auto", "none"}:
        return ""
    aliases = {
        "together": "together_ai",
    }
    return aliases.get(normalized, normalized)


def get_repo_root(repo_root: Path | str | None = None) -> Path:
    return Path(repo_root) if repo_root else _REPO_ROOT


def get_pipeline_config_path(repo_root: Path | str | None = None) -> Path:
    return get_repo_root(repo_root) / _PIPELINE_CONFIG_NAME


def _load_pipeline_raw(repo_root: Path | str | None = None) -> Dict[str, Any]:
    path = get_pipeline_config_path(repo_root)
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_pipeline_raw(data: Dict[str, Any], repo_root: Path | str | None = None) -> Path:
    path = get_pipeline_config_path(repo_root)
    path.write_text(
        yaml.safe_dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _detect_provider_from_endpoint(endpoint: str) -> str | None:
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


def _parse_n_clusters(raw: Dict[str, Any]) -> int:
    val = raw.get("n_clusters", 7)
    try:
        return int(val)
    except Exception:
        return 7


def load_pipeline_fields(repo_root: Path | str | None = None) -> Dict[str, Any]:
    """
    Load UI-relevant fields from pipeline_config.yaml.

    Returns normalized keys used by Streamlit settings.
    """
    raw = _load_pipeline_raw(repo_root)

    api_endpoint = str(raw.get("api_endpoint") or raw.get("azure_endpoint") or "")
    api_provider = _normalize_provider(str(raw.get("api_provider") or ""))
    if not api_provider:
        api_provider = _detect_provider_from_endpoint(api_endpoint) or ""

    api_model = str(raw.get("api_model") or raw.get("api_deployment") or raw.get("azure_deployment") or "")
    return {
        "data_path": str(raw.get("data_path") or ""),
        "sample_id": str(raw.get("sample_id") or ""),
        "n_clusters": _parse_n_clusters(raw),
        "api_provider": api_provider,
        "api_key": str(raw.get("api_key") or raw.get("azure_openai_key") or ""),
        "api_endpoint": api_endpoint,
        "api_version": str(raw.get("api_version") or raw.get("azure_api_version") or ""),
        "api_model": api_model,
        "api_deployment": api_model,
    }


def save_pipeline_fields(
    *,
    repo_root: Path | str | None = None,
    **kwargs: Any,
) -> Path:
    """
    Persist selected fields to pipeline_config.yaml.

    Supports keys:
      data_path, sample_id, n_clusters, api_provider, api_key, api_endpoint, api_version, api_model, api_deployment
    """
    allowed = {
        "data_path",
        "sample_id",
        "n_clusters",
        "api_provider",
        "api_key",
        "api_endpoint",
        "api_version",
        "api_model",
        "api_deployment",
    }

    raw = _load_pipeline_raw(repo_root)
    for key, value in kwargs.items():
        if key not in allowed:
            continue
        if value is None:
            continue
        if key == "api_provider":
            raw[key] = _normalize_provider(str(value))
        elif key == "n_clusters":
            try:
                raw[key] = int(value)
            except Exception:
                continue
        else:
            raw[key] = str(value).strip()

    # Normalize model/deployment aliases to the canonical api_model.
    api_model = raw.get("api_model")
    api_deployment = raw.get("api_deployment")
    if (api_model is None or str(api_model).strip() == "") and api_deployment is not None:
        api_model = str(api_deployment).strip()
    if api_model is not None:
        canonical = str(api_model).strip()
        raw["api_model"] = canonical
        raw["api_deployment"] = canonical

    provider = _normalize_provider(str(raw.get("api_provider") or ""))
    raw["api_provider"] = provider
    endpoint = str(raw.get("api_endpoint") or "")
    effective_provider = provider or _detect_provider_from_endpoint(endpoint) or ""
    if effective_provider == "azure":
        raw["azure_openai_key"] = str(raw.get("api_key") or "")
        raw["azure_endpoint"] = str(raw.get("api_endpoint") or "")
        raw["azure_api_version"] = str(raw.get("api_version") or "")
        raw["azure_deployment"] = str(raw.get("api_model") or raw.get("api_deployment") or "")
    else:
        # Keep pipeline_config canonical across providers; avoid stale Azure mirrors.
        for k in ("azure_openai_key", "azure_endpoint", "azure_api_version", "azure_deployment"):
            raw.pop(k, None)

    return _save_pipeline_raw(raw, repo_root)
