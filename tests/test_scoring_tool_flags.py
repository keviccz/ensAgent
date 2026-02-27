from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ensagent_tools.config_manager import PipelineConfig
from ensagent_tools.scoring import run_scoring


class ScoringToolFlagTests(unittest.TestCase):
    def test_run_scoring_uses_cfg_vlm_off_by_default(self) -> None:
        cfg = PipelineConfig(vlm_off=True)
        with patch("ensagent_tools.scoring.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0)
            run_scoring(cfg)

        cmd = mock_run.call_args[0][0]
        self.assertIn("--vlm_off", cmd)

    def test_run_scoring_respects_explicit_override(self) -> None:
        cfg = PipelineConfig(vlm_off=True)
        with patch("ensagent_tools.scoring.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0)
            run_scoring(cfg, vlm_off=False)

        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--vlm_off", cmd)


if __name__ == "__main__":
    unittest.main()
