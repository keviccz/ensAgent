from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ensagent_tools.config_manager import PipelineConfig
from ensagent_tools.scoring import run_scoring


class ScoringToolFlagTests(unittest.TestCase):
    def test_run_scoring_uses_cfg_vlm_off_by_default(self) -> None:
        cfg = PipelineConfig(vlm_off=True, temperature=0.25, top_p=0.9)
        with patch("ensagent_tools.scoring.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0)
            run_scoring(cfg)

        cmd = mock_run.call_args[0][0]
        self.assertIn("--vlm_off", cmd)
        self.assertIn("--temperature", cmd)
        self.assertIn("0.25", cmd)
        self.assertIn("--top_p", cmd)
        self.assertIn("0.9", cmd)

    def test_run_scoring_respects_explicit_override(self) -> None:
        cfg = PipelineConfig(vlm_off=True)
        with patch("ensagent_tools.scoring.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0)
            run_scoring(cfg, vlm_off=False)

        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--vlm_off", cmd)

    def test_run_scoring_respects_sampling_overrides(self) -> None:
        cfg = PipelineConfig(temperature=0.7, top_p=1.0)
        with patch("ensagent_tools.scoring.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0)
            run_scoring(cfg, temperature=0.1, top_p=0.55)

        cmd = mock_run.call_args[0][0]
        temp_idx = cmd.index("--temperature")
        top_p_idx = cmd.index("--top_p")
        self.assertEqual(cmd[temp_idx + 1], "0.1")
        self.assertEqual(cmd[top_p_idx + 1], "0.55")

    def test_run_scoring_passes_provider_arguments(self) -> None:
        cfg = PipelineConfig(
            api_provider="openrouter",
            api_key="k-test",
            api_endpoint="https://openrouter.ai/api/v1",
            api_version="",
            api_model="openai/gpt-4o-mini",
        )
        with patch("ensagent_tools.scoring.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0)
            run_scoring(cfg)

        cmd = mock_run.call_args[0][0]
        self.assertIn("--api_provider", cmd)
        self.assertIn("openrouter", cmd)
        self.assertIn("--api_key", cmd)
        self.assertIn("k-test", cmd)
        self.assertIn("--api_endpoint", cmd)
        self.assertIn("https://openrouter.ai/api/v1", cmd)
        self.assertIn("--api_model", cmd)
        self.assertIn("openai/gpt-4o-mini", cmd)


if __name__ == "__main__":
    unittest.main()
