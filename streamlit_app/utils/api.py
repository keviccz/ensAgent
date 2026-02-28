"""
API wrapper for LLM integration with robust error handling.
Supports LiteLLM unified multi-provider routing, with OpenAI/Anthropic fallback.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import streamlit as st


class APIError(Exception):
    """Custom exception for API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class ThinkingStep:
    """Represents a thinking/reasoning step from the model."""
    step_type: str  # "thinking", "tool_call", "tool_result", "output"
    content: str
    metadata: Dict[str, Any] = None


def _detect_provider(endpoint: str) -> str | None:
    """Auto-detect API provider from endpoint URL."""
    if not endpoint:
        return None
    endpoint_lower = endpoint.lower().strip()

    if "openai.azure.com" in endpoint_lower or "cognitiveservices.azure.com" in endpoint_lower:
        return "azure"
    elif "api.openai.com" in endpoint_lower:
        return "openai"
    elif "api.anthropic.com" in endpoint_lower or "anthropic" in endpoint_lower:
        return "anthropic"
    elif "openrouter.ai" in endpoint_lower:
        return "openrouter"
    elif "api.deepseek.com" in endpoint_lower:
        return "deepseek"
    elif "api.groq.com" in endpoint_lower:
        return "groq"
    elif "api.together.xyz" in endpoint_lower:
        return "together_ai"
    elif "api.mistral.ai" in endpoint_lower:
        return "mistral"
    elif "api.cohere.com" in endpoint_lower:
        return "cohere"
    elif "api.x.ai" in endpoint_lower:
        return "xai"
    elif "api.perplexity.ai" in endpoint_lower:
        return "perplexity"
    elif "generativelanguage.googleapis.com" in endpoint_lower:
        return "gemini"

    return None


def _normalize_provider(provider: str | None) -> str | None:
    """Normalize legacy and UI provider values to runtime provider taxonomy."""
    if provider is None:
        return None
    normalized = str(provider).strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    alias_map = {
        "together": "together_ai",
    }
    return alias_map.get(normalized, normalized)


def _read_pipeline_api_credentials(repo_root: Path | None = None) -> Dict[str, str]:
    """Read normalized API credentials from pipeline_config.yaml."""
    try:
        from streamlit_app.utils.config_bridge import load_pipeline_fields

        fields = load_pipeline_fields(repo_root=repo_root)
        endpoint = fields.get("api_endpoint", "")
        provider = _normalize_provider(fields.get("api_provider")) or _detect_provider(endpoint) or ""
        return {
            "api_provider": provider,
            "api_key": fields.get("api_key", ""),
            "api_endpoint": endpoint,
            "api_version": fields.get("api_version", ""),
            "api_model": fields.get("api_model") or fields.get("api_deployment", ""),
            "api_deployment": fields.get("api_deployment", ""),
        }
    except Exception:
        return {
            "api_provider": "",
            "api_key": "",
            "api_endpoint": "",
            "api_version": "",
            "api_model": "",
            "api_deployment": "",
        }


def _get_api_credentials() -> tuple[str, str, str, str, str]:
    """
    Get API credentials from various sources.
    Returns: (api_key, endpoint, api_version, model, provider)
    """
    # Priority: session_state > pipeline_config.yaml > env vars > config files
    api_key = st.session_state.get("api_key") or ""
    endpoint = st.session_state.get("api_endpoint") or ""
    provider = _normalize_provider(st.session_state.get("api_provider"))
    model = st.session_state.get("model_name") or ""
    api_version = st.session_state.get("api_version") or ""
    model_override = st.session_state.get("api_model") or ""
    deployment = st.session_state.get("api_deployment") or ""
    # Use deployment name as model if provided.
    if model_override:
        model = model_override
    elif deployment:
        model = deployment

    pipeline_creds = _read_pipeline_api_credentials()
    if not provider:
        provider = _normalize_provider(pipeline_creds.get("api_provider"))
    if not api_key:
        api_key = pipeline_creds.get("api_key", "")
    if not endpoint:
        endpoint = pipeline_creds.get("api_endpoint", "")
    if not api_version:
        api_version = pipeline_creds.get("api_version", "")
    if not model:
        model = pipeline_creds.get("api_model", "")
    if not model:
        model = pipeline_creds.get("api_deployment", "")

    enforce_custom_endpoint = provider in {"others", "openai_compatible"}

    # Try environment variables
    if not api_key:
        api_key = next(
            (
                os.environ.get(k)
                for k in [
                    "AZURE_OPENAI_KEY",
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "GOOGLE_API_KEY",
                    "GEMINI_API_KEY",
                    "OPENROUTER_API_KEY",
                    "DEEPSEEK_API_KEY",
                    "GROQ_API_KEY",
                    "TOGETHER_API_KEY",
                    "MISTRAL_API_KEY",
                    "COHERE_API_KEY",
                    "XAI_API_KEY",
                    "PERPLEXITY_API_KEY",
                ]
                if os.environ.get(k)
            ),
            "",
        )
    if not endpoint and not enforce_custom_endpoint:
        endpoint = (
            os.environ.get("AZURE_OPENAI_ENDPOINT") or
            os.environ.get("OPENAI_API_BASE") or
            ""
        )

    # Try loading from config files
    if not api_key or (not endpoint and not enforce_custom_endpoint):
        try:
            repo_root = Path(__file__).parent.parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))

            # Try api_config.py first
            try:
                import api_config as _api_cfg  # type: ignore
                warnings.warn(
                    "api_config.py is deprecated; prefer pipeline_config.yaml or environment variables.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if not api_key:
                    api_key = getattr(_api_cfg, "AZURE_OPENAI_KEY", "")
                if not endpoint:
                    endpoint = (
                        getattr(_api_cfg, "AZURE_OPENAI_ENDPOINT", "")
                        or getattr(_api_cfg, "AZURE_ENDPOINT", "")
                    )
                if not api_version:
                    api_version = (
                        getattr(_api_cfg, "AZURE_OPENAI_API_VERSION", "")
                        or getattr(_api_cfg, "AZURE_API_VERSION", "")
                    )
                if not model:
                    model = (
                        getattr(_api_cfg, "AZURE_OPENAI_DEPLOYMENT", "")
                        or getattr(_api_cfg, "AZURE_DEPLOYMENT", "")
                    )
            except ImportError:
                pass

            # Try scoring/config.py
            try:
                from scoring.config import (
                    AZURE_OPENAI_KEY as SCORING_KEY,
                    AZURE_ENDPOINT as SCORING_ENDPOINT,
                    AZURE_API_VERSION,
                    AZURE_DEPLOYMENT,
                )
                if not api_key:
                    api_key = SCORING_KEY
                if not endpoint:
                    endpoint = SCORING_ENDPOINT
                if not api_version:
                    api_version = AZURE_API_VERSION
                if not model:
                    model = AZURE_DEPLOYMENT
                if not provider:
                    provider = "azure"
            except ImportError:
                pass
        except Exception:
            pass

    # Auto-detect provider from endpoint when not explicitly set
    if not provider:
        provider = _detect_provider(endpoint) or "openai"

    # Set default endpoints for providers
    if not endpoint:
        default_endpoints = {
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
        endpoint = default_endpoints.get(provider, endpoint)

    # Use session state api_version, fallback to env var
    if not api_version:
        api_version = (
            os.environ.get("AZURE_OPENAI_API_VERSION")
            or os.environ.get("AZURE_API_VERSION")
            or "2024-12-01-preview"
        )
    if not model:
        model = (
            os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or os.environ.get("AZURE_DEPLOYMENT")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4o"
        )

    return api_key, endpoint, api_version, model, provider or "openai"


class EnsAgentAPI:
    """
    API client for EnsAgent LLM integration.
    Supports OpenAI-compatible and Anthropic with function calling.
    """
    
    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "",
        model: str = "",
        provider: str = "",
        api_version: str = "",
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.provider = provider
        self.api_version = api_version
        self._client = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the API client."""
        if self._initialized:
            return True

        self.provider = _normalize_provider(self.provider)

        provider_needs_endpoint = self.provider in {"others", "openai_compatible", "azure"}
        has_explicit_credentials = bool(
            self.api_key
            and self.model
            and self.provider
            and (self.endpoint or not provider_needs_endpoint)
        )
        if not has_explicit_credentials:
            key, endpoint, api_version, model, provider = _get_api_credentials()

            if not self.api_key:
                self.api_key = key
            if not self.endpoint:
                self.endpoint = endpoint
            if not self.model:
                self.model = model
            if not self.provider:
                self.provider = _normalize_provider(provider)
            if not self.api_version:
                self.api_version = api_version

        if not self.provider:
            self.provider = "openai"
        if not self.api_version:
            self.api_version = (
                os.environ.get("AZURE_OPENAI_API_VERSION")
                or os.environ.get("AZURE_API_VERSION")
                or "2024-12-01-preview"
            )

        # Custom OpenAI-compatible mode requires endpoint.
        if not self.api_key:
            return False
        if self.provider in {"others", "openai_compatible"} and not self.endpoint:
            return False

        try:
            from litellm import completion  # type: ignore

            self._client = completion
            self._client_type = "litellm"
            self._initialized = True
            return True
        except ImportError:
            pass

        try:
            if self.provider == "anthropic":
                self._init_anthropic()
            elif self.provider == "azure" or _detect_provider(self.endpoint) == "azure":
                self._init_azure(self.api_version)
            else:
                self._init_openai()

            self._initialized = True
            return True
        except ImportError as e:
            raise APIError(f"Missing package: {e}. Install with pip install litellm openai anthropic")
        except Exception as e:
            raise APIError(f"Failed to initialize API client: {e}")
    
    def _init_azure(self, api_version: str) -> None:
        """Initialize Azure OpenAI client."""
        from openai import AzureOpenAI
        self._client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=api_version,
        )
        self._client_type = "openai"
    
    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.endpoint if self.endpoint else None,
        )
        self._client_type = "openai"
    
    def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
            )
            self._client_type = "anthropic"
        except ImportError:
            raise APIError("anthropic package not installed. Run: pip install anthropic")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions for function calling.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            max_tokens: Maximum tokens in response.
            stream: Whether to stream the response.
        
        Returns:
            Response dict with 'content', 'tool_calls', and 'thinking' keys.
        """
        if not self._initialized:
            if not self.initialize():
                raise APIError("API not initialized. Please configure API credentials.")
        
        try:
            if self._client_type == "litellm":
                return self._chat_litellm(messages, tools, temperature, top_p, max_tokens)
            if self._client_type == "anthropic":
                return self._chat_anthropic(messages, tools, temperature, top_p, max_tokens)
            else:  # openai or azure
                return self._chat_openai(messages, tools, temperature, top_p, max_tokens, stream)
        
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise APIError(str(e), status_code=429, retryable=True)
            elif "authentication" in error_str or "401" in error_str:
                raise APIError(str(e), status_code=401, retryable=False)
            else:
                raise APIError(str(e))
    
    def _chat_litellm(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Chat using LiteLLM unified providers."""
        resolved_provider = _normalize_provider(self.provider) or _detect_provider(self.endpoint) or "openai"
        model = self._resolve_litellm_model(provider=resolved_provider, model=self.model)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.endpoint and resolved_provider in {
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
            kwargs["api_base"] = self.endpoint
        if self.api_version and resolved_provider == "azure":
            kwargs["api_version"] = self.api_version

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client(**kwargs)
        return self._parse_litellm_response(response)

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stream: bool,
    ) -> Dict[str, Any]:
        """Chat using OpenAI/Azure client."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        response = self._client.chat.completions.create(**kwargs)
        return self._parse_openai_response(response)
    
    def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Chat using Anthropic client."""
        # Convert messages format for Anthropic
        system_msg = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        
        kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        
        if system_msg:
            kwargs["system"] = system_msg
        
        if tools:
            # Convert OpenAI tool format to Anthropic format
            anthropic_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    anthropic_tools.append({
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                    })
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools
        
        response = self._client.messages.create(**kwargs)
        return self._parse_anthropic_response(response)
    
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse OpenAI/Azure API response."""
        choice = response.choices[0]
        message = choice.message
        
        result = {
            "content": message.content or "",
            "tool_calls": [],
            "thinking": [],
            "finish_reason": choice.finish_reason,
        }
        
        if message.tool_calls:
            for tc in message.tool_calls:
                raw_args = tc.function.arguments or "{}"
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed_args = {"raw": raw_args}
                else:
                    parsed_args = raw_args
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": parsed_args,
                })
        
        return result
    
    def _parse_anthropic_response(self, response) -> Dict[str, Any]:
        """Parse Anthropic API response."""
        result = {
            "content": "",
            "tool_calls": [],
            "thinking": [],
            "finish_reason": response.stop_reason,
        }
        
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })
        
        return result

    def _parse_litellm_response(self, response: Any) -> Dict[str, Any]:
        """Parse LiteLLM response into internal normalized shape."""
        if hasattr(response, "model_dump"):
            payload = response.model_dump()
        elif isinstance(response, dict):
            payload = response
        else:
            payload = json.loads(str(response))

        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        tool_calls = message.get("tool_calls") or []
        parsed_tool_calls = []
        for tc in tool_calls:
            function = tc.get("function", {})
            raw_args = function.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"raw": raw_args}
            else:
                args = raw_args
            parsed_tool_calls.append(
                {
                    "id": tc.get("id"),
                    "name": function.get("name"),
                    "arguments": args,
                }
            )

        return {
            "content": message.get("content") or "",
            "tool_calls": parsed_tool_calls,
            "thinking": [],
            "finish_reason": choice.get("finish_reason"),
        }

    def _resolve_litellm_model(self, *, provider: str, model: str) -> str:
        """Resolve provider/model to a LiteLLM routing model string."""
        cleaned_model = (model or "").strip()
        if not cleaned_model:
            cleaned_model = "gpt-4o"
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
    
    def with_retry(
        self,
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> Any:
        """
        Execute a function with exponential backoff retry.
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except APIError as e:
                last_error = e
                if not e.retryable or attempt >= max_retries:
                    raise
                
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
        
        raise last_error


# Tool definitions imported from the unified ensagent_tools registry.
# This ensures the schemas stay in sync across CLI agent, UI, and programmatic usage.
try:
    from ensagent_tools.registry import TOOL_SCHEMAS as ENSAGENT_TOOLS
except ImportError:
    ENSAGENT_TOOLS = []
