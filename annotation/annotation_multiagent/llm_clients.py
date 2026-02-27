from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import openai
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type


def make_azure_openai_client(
    azure_endpoint: str,
    api_key: str,
    api_version: str,
) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )


@retry(
    reraise=True,
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type((openai.OpenAIError, Exception)),
)
def chat_json(
    client: AzureOpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Call Azure OpenAI and parse response as JSON dict. Raises on parse failure."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = resp.choices[0].message.content or ""
    cleaned = text.strip()
    # tolerate code fences
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data




















