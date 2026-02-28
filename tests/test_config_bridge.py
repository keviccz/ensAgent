from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml


class ConfigBridgeTests(unittest.TestCase):
    def test_save_and_load_pipeline_fields(self) -> None:
        from streamlit_app.utils.config_bridge import load_pipeline_fields, save_pipeline_fields

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            save_pipeline_fields(
                repo_root=repo,
                api_provider="azure",
                api_key="test-key",
                api_endpoint="https://example.openai.azure.com/",
                api_version="2024-12-01-preview",
                api_model="gpt-4o",
                api_deployment="gpt-4o",
                data_path="D:/data/151507",
                sample_id="DLPFC_151507",
                n_clusters=9,
            )

            loaded = load_pipeline_fields(repo_root=repo)
            self.assertEqual(loaded["api_provider"], "azure")
            self.assertEqual(loaded["api_key"], "test-key")
            self.assertEqual(loaded["api_endpoint"], "https://example.openai.azure.com/")
            self.assertEqual(loaded["api_version"], "2024-12-01-preview")
            self.assertEqual(loaded["api_model"], "gpt-4o")
            self.assertEqual(loaded["api_deployment"], "gpt-4o")
            self.assertEqual(loaded["data_path"], "D:/data/151507")
            self.assertEqual(loaded["sample_id"], "DLPFC_151507")
            self.assertEqual(loaded["n_clusters"], 9)

            raw = yaml.safe_load((repo / "pipeline_config.yaml").read_text(encoding="utf-8")) or {}
            self.assertEqual(raw.get("azure_openai_key"), "test-key")
            self.assertEqual(raw.get("azure_endpoint"), "https://example.openai.azure.com/")
            self.assertEqual(raw.get("azure_api_version"), "2024-12-01-preview")
            self.assertEqual(raw.get("azure_deployment"), "gpt-4o")
            self.assertEqual(raw.get("n_clusters"), 9)

    def test_n_clusters_is_saved_as_integer(self) -> None:
        from streamlit_app.utils.config_bridge import load_pipeline_fields, save_pipeline_fields

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            save_pipeline_fields(repo_root=repo, n_clusters="11")

            raw = yaml.safe_load((repo / "pipeline_config.yaml").read_text(encoding="utf-8")) or {}
            self.assertEqual(raw.get("n_clusters"), 11)
            self.assertIsInstance(raw.get("n_clusters"), int)

            loaded = load_pipeline_fields(repo_root=repo)
            self.assertEqual(loaded["n_clusters"], 11)

    def test_sampling_parameters_are_saved_as_floats(self) -> None:
        from streamlit_app.utils.config_bridge import load_pipeline_fields, save_pipeline_fields

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            save_pipeline_fields(repo_root=repo, temperature="0.35", top_p="0.82")

            raw = yaml.safe_load((repo / "pipeline_config.yaml").read_text(encoding="utf-8")) or {}
            self.assertEqual(raw.get("temperature"), 0.35)
            self.assertEqual(raw.get("top_p"), 0.82)

            loaded = load_pipeline_fields(repo_root=repo)
            self.assertEqual(loaded["temperature"], 0.35)
            self.assertEqual(loaded["top_p"], 0.82)

    def test_sampling_parameters_are_normalized_without_float_noise(self) -> None:
        from streamlit_app.utils.config_bridge import save_pipeline_fields

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            save_pipeline_fields(repo_root=repo, temperature=0.9999999999999999)

            raw = yaml.safe_load((repo / "pipeline_config.yaml").read_text(encoding="utf-8")) or {}
            self.assertEqual(raw.get("temperature"), 1.0)

    def test_load_uses_azure_fallback_keys(self) -> None:
        from streamlit_app.utils.config_bridge import load_pipeline_fields

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "pipeline_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "azure_openai_key": "k",
                        "azure_endpoint": "https://fallback.openai.azure.com/",
                        "azure_api_version": "2024-12-01-preview",
                        "azure_deployment": "gpt-4o",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            loaded = load_pipeline_fields(repo_root=repo)
            self.assertEqual(loaded["api_key"], "k")
            self.assertEqual(loaded["api_endpoint"], "https://fallback.openai.azure.com/")
            self.assertEqual(loaded["api_version"], "2024-12-01-preview")
            self.assertEqual(loaded["api_model"], "gpt-4o")
            self.assertEqual(loaded["api_deployment"], "gpt-4o")

    def test_load_uses_api_deployment_as_model_alias(self) -> None:
        from streamlit_app.utils.config_bridge import load_pipeline_fields

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "pipeline_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "api_provider": "anthropic",
                        "api_key": "k",
                        "api_endpoint": "https://api.anthropic.com",
                        "api_deployment": "claude-3-5-sonnet-latest",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            loaded = load_pipeline_fields(repo_root=repo)
            self.assertEqual(loaded["api_model"], "claude-3-5-sonnet-latest")
            self.assertEqual(loaded["api_deployment"], "claude-3-5-sonnet-latest")

    def test_save_non_azure_provider_removes_stale_azure_alias_fields(self) -> None:
        from streamlit_app.utils.config_bridge import save_pipeline_fields

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "pipeline_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "api_provider": "azure",
                        "azure_openai_key": "old-key",
                        "azure_endpoint": "https://old.openai.azure.com/",
                        "azure_api_version": "2023-12-01-preview",
                        "azure_deployment": "old-deployment",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            save_pipeline_fields(
                repo_root=repo,
                api_provider="Anthropic",
                api_key="anthropic-key",
                api_endpoint="https://api.anthropic.com",
                api_model="claude-3-5-sonnet-latest",
            )

            raw = yaml.safe_load((repo / "pipeline_config.yaml").read_text(encoding="utf-8")) or {}
            self.assertEqual(raw.get("api_provider"), "anthropic")
            self.assertEqual(raw.get("api_model"), "claude-3-5-sonnet-latest")
            self.assertEqual(raw.get("api_deployment"), "claude-3-5-sonnet-latest")
            self.assertNotIn("azure_openai_key", raw)
            self.assertNotIn("azure_endpoint", raw)
            self.assertNotIn("azure_api_version", raw)
            self.assertNotIn("azure_deployment", raw)


if __name__ == "__main__":
    unittest.main()
