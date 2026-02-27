from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_pic_config_module():
    repo = Path(__file__).resolve().parent.parent
    module_path = repo / "scoring" / "pic_analyze" / "config.py"
    spec = importlib.util.spec_from_file_location("pic_analyze_config_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PicAnalyzeConfigTests(unittest.TestCase):
    def test_pipeline_generic_azure_fields_are_supported(self) -> None:
        mod = _load_pic_config_module()
        resolved = mod._resolve_azure_openai_settings(
            env={},
            pipeline_raw={
                "api_provider": "azure",
                "api_key": "k1",
                "api_endpoint": "https://example.openai.azure.com/",
                "api_version": "2024-12-01-preview",
                "api_model": "gpt-4.1-mini",
            },
        )

        self.assertTrue(resolved["is_azure_provider"])
        self.assertEqual(resolved["api_key"], "k1")
        self.assertEqual(resolved["endpoint"], "https://example.openai.azure.com/")
        self.assertEqual(resolved["deployment_name"], "gpt-4.1-mini")
        self.assertEqual(resolved["ocr_deployment_name"], "gpt-4.1-mini")

    def test_env_overrides_pipeline_values(self) -> None:
        mod = _load_pic_config_module()
        resolved = mod._resolve_azure_openai_settings(
            env={
                "AZURE_OPENAI_API_KEY": "env-key",
                "AZURE_OPENAI_DEPLOYMENT_NAME": "env-model",
                "AZURE_OPENAI_OCR_DEPLOYMENT_NAME": "ocr-model",
            },
            pipeline_raw={
                "api_provider": "azure",
                "api_key": "pipeline-key",
                "api_model": "pipeline-model",
            },
        )

        self.assertEqual(resolved["api_key"], "env-key")
        self.assertEqual(resolved["deployment_name"], "env-model")
        self.assertEqual(resolved["ocr_deployment_name"], "ocr-model")

    def test_non_azure_provider_is_detected(self) -> None:
        mod = _load_pic_config_module()
        resolved = mod._resolve_azure_openai_settings(
            env={},
            pipeline_raw={
                "api_provider": "anthropic",
                "api_endpoint": "https://api.anthropic.com",
            },
        )

        self.assertFalse(resolved["is_azure_provider"])
        self.assertEqual(resolved["provider"], "anthropic")


if __name__ == "__main__":
    unittest.main()
