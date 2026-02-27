import unittest

from streamlit_app.components import settings as settings_mod
from streamlit_app.utils import api as api_mod
import tempfile
from pathlib import Path
import yaml


class ProviderConfigurationTests(unittest.TestCase):
    def test_settings_provider_catalog_includes_openrouter(self) -> None:
        labels = [label for label, _ in settings_mod.get_provider_catalog()]
        self.assertIn("OpenRouter", labels)

    def test_default_endpoint_for_groq(self) -> None:
        self.assertEqual(settings_mod._get_default_endpoint("groq"), "https://api.groq.com/openai/v1")

    def test_detect_provider_openrouter(self) -> None:
        detected = api_mod._detect_provider("https://openrouter.ai/api/v1")
        self.assertEqual(detected, "openrouter")

    def test_read_pipeline_api_credentials_prefers_generic_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "pipeline_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "api_provider": "azure",
                        "api_key": "k1",
                        "api_endpoint": "https://generic.openai.azure.com/",
                        "api_version": "2024-12-01-preview",
                        "api_model": "gpt-4o",
                        "api_deployment": "gpt-4o",
                        "azure_openai_key": "legacy-k",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            creds = api_mod._read_pipeline_api_credentials(repo_root=repo)
            self.assertEqual(creds["api_provider"], "azure")
            self.assertEqual(creds["api_key"], "k1")
            self.assertEqual(creds["api_endpoint"], "https://generic.openai.azure.com/")
            self.assertEqual(creds["api_version"], "2024-12-01-preview")
            self.assertEqual(creds["api_model"], "gpt-4o")
            self.assertEqual(creds["api_deployment"], "gpt-4o")

    def test_read_pipeline_api_credentials_falls_back_to_azure_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "pipeline_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "azure_openai_key": "legacy-k",
                        "azure_endpoint": "https://legacy.openai.azure.com/",
                        "azure_api_version": "2024-12-01-preview",
                        "azure_deployment": "gpt-4o",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            creds = api_mod._read_pipeline_api_credentials(repo_root=repo)
            self.assertEqual(creds["api_provider"], "azure")
            self.assertEqual(creds["api_key"], "legacy-k")
            self.assertEqual(creds["api_endpoint"], "https://legacy.openai.azure.com/")
            self.assertEqual(creds["api_version"], "2024-12-01-preview")
            self.assertEqual(creds["api_model"], "gpt-4o")
            self.assertEqual(creds["api_deployment"], "gpt-4o")


if __name__ == "__main__":
    unittest.main()
