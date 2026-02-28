from __future__ import annotations

from typing import Any, Dict, List

from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

try:
    from provider_runtime import ProviderConfig, resolve_provider_config, completion_json
except Exception:
    from scoring.provider_runtime import ProviderConfig, resolve_provider_config, completion_json  # type: ignore


def make_provider_client(
    *,
    api_provider: str = "",
    api_key: str = "",
    api_endpoint: str = "",
    api_version: str = "",
    api_model: str = "",
    azure_openai_key: str = "",
    azure_endpoint: str = "",
    azure_api_version: str = "",
    azure_deployment: str = "",
) -> Dict[str, Any]:
    config = resolve_provider_config(
        api_provider=api_provider,
        api_key=api_key,
        api_endpoint=api_endpoint,
        api_version=api_version,
        api_model=api_model,
        azure_openai_key=azure_openai_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )
    return {"provider_config": config}


def make_azure_openai_client(
    azure_endpoint: str,
    api_key: str,
    api_version: str,
) -> Dict[str, Any]:
    """Backward-compatible constructor name; now returns a generic provider client."""
    return make_provider_client(
        api_provider="azure",
        api_key=api_key,
        api_endpoint=azure_endpoint,
        api_version=api_version,
    )


@retry(
    reraise=True,
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type((Exception,)),
)
def chat_json(
    client: Dict[str, Any],
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Call configured provider and parse response as JSON dict."""
    cfg = client.get("provider_config")
    if not isinstance(cfg, ProviderConfig):
        raise ValueError("Invalid provider client: missing provider_config")
    return completion_json(
        config=cfg,
        model=model or cfg.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
    )

