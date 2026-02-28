from __future__ import annotations

import unittest

try:
    from provider_runtime import resolve_provider_config, resolve_litellm_model, detect_provider
except Exception:
    from scoring.provider_runtime import resolve_provider_config, resolve_litellm_model, detect_provider  # type: ignore


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


if __name__ == "__main__":
    unittest.main()

