from __future__ import annotations

import types
import unittest
from unittest.mock import patch

try:
    from provider_runtime import (
        resolve_provider_config,
        resolve_litellm_model,
        detect_provider,
        completion_raw,
        ProviderConfig,
    )
except Exception:
    from scoring.provider_runtime import (  # type: ignore
        resolve_provider_config,
        resolve_litellm_model,
        detect_provider,
        completion_raw,
        ProviderConfig,
    )


class ProviderRuntimeTests(unittest.TestCase):
    def test_detect_provider_from_endpoint(self) -> None:
        self.assertEqual(detect_provider("https://foo.openai.azure.com/"), "azure")
        self.assertEqual(detect_provider("https://openrouter.ai/api/v1"), "openrouter")
        self.assertEqual(detect_provider("https://api.openai.com/v1"), "openai")

    def test_resolve_provider_prefers_explicit_provider(self) -> None:
        cfg = resolve_provider_config(
            api_provider="anthropic",
            api_key="k",
            api_model="claude-3-5-sonnet-latest",
        )
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.api_key, "k")
        self.assertEqual(cfg.model, "claude-3-5-sonnet-latest")

    def test_resolve_provider_azure_alias_fallback(self) -> None:
        cfg = resolve_provider_config(
            api_provider="azure",
            azure_openai_key="k-az",
            azure_endpoint="https://r.openai.azure.com/",
            azure_api_version="2024-12-01-preview",
            azure_deployment="gpt-4o",
        )
        self.assertEqual(cfg.provider, "azure")
        self.assertEqual(cfg.api_key, "k-az")
        self.assertEqual(cfg.endpoint, "https://r.openai.azure.com/")
        self.assertEqual(cfg.model, "gpt-4o")

    def test_litellm_model_prefix(self) -> None:
        self.assertEqual(resolve_litellm_model(provider="anthropic", model="claude-3-5-sonnet-latest"), "anthropic/claude-3-5-sonnet-latest")
        self.assertEqual(resolve_litellm_model(provider="openai", model="gpt-4o"), "gpt-4o")

    def test_completion_raw_falls_back_to_openai_sdk_without_litellm(self) -> None:
        calls: dict[str, object] = {}

        class _DummyCompletions:
            @staticmethod
            def create(**kwargs):
                calls["kwargs"] = kwargs
                return {"choices": [{"message": {"content": "ok"}}]}

        class _DummyChat:
            completions = _DummyCompletions()

        class _DummyOpenAI:
            def __init__(self, api_key=None, base_url=None):
                calls["api_key"] = api_key
                calls["base_url"] = base_url
                self.chat = _DummyChat()

        fake_openai_module = types.SimpleNamespace(OpenAI=_DummyOpenAI)
        cfg = ProviderConfig(
            provider="openai",
            api_key="k-test",
            endpoint="https://api.openai.com/v1",
            api_version="",
            model="gpt-4o-mini",
            ocr_model="gpt-4o-mini",
        )

        with patch.dict("sys.modules", {"litellm": None, "openai": fake_openai_module}):
            out = completion_raw(
                config=cfg,
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-4o-mini",
                temperature=0.2,
                top_p=0.9,
                max_tokens=32,
            )

        self.assertIn("choices", out)
        self.assertEqual(calls["api_key"], "k-test")
        self.assertEqual(calls["base_url"], "https://api.openai.com/v1")
        sent = calls["kwargs"]
        self.assertIsInstance(sent, dict)
        self.assertEqual(sent["model"], "gpt-4o-mini")
        self.assertEqual(sent["max_tokens"], 32)


if __name__ == "__main__":
    unittest.main()
