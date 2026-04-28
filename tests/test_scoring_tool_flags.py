from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import ensagent_tools.scoring as scoring_mod
from ensagent_tools.config_manager import PipelineConfig
from ensagent_tools.scoring import run_scoring


class ScoringToolFlagTests(unittest.TestCase):
    def test_run_scoring_uses_cfg_vlm_off_by_default(self) -> None:
        cfg = PipelineConfig(vlm_off=True, temperature=0.25, top_p=0.9)
        with patch("ensagent_tools.scoring.Path.exists", return_value=True), patch("ensagent_tools.scoring.subprocess.run") as mock_run:
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
        with patch("ensagent_tools.scoring.Path.exists", return_value=True), patch("ensagent_tools.scoring.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0)
            run_scoring(cfg, vlm_off=False)

        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--vlm_off", cmd)

    def test_run_scoring_respects_sampling_overrides(self) -> None:
        cfg = PipelineConfig(temperature=0.7, top_p=1.0)
        with patch("ensagent_tools.scoring.Path.exists", return_value=True), patch("ensagent_tools.scoring.subprocess.run") as mock_run:
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
        with patch("ensagent_tools.scoring.Path.exists", return_value=True), patch("ensagent_tools.scoring.subprocess.run") as mock_run:
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

    def test_run_scoring_uses_cfg_csv_path_by_default(self) -> None:
        cfg = PipelineConfig(csv_path="custom/input")
        with patch("ensagent_tools.scoring.Path.exists", return_value=True), patch("ensagent_tools.scoring.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0)
            run_scoring(cfg)

        cmd = mock_run.call_args[0][0]
        self.assertIn("--input_dir", cmd)
        input_idx = cmd.index("--input_dir")
        self.assertEqual(str(cmd[input_idx + 1]).replace("\\", "/"), "custom/input")

    def test_run_scoring_autoruns_pic_analyze_when_visual_file_missing(self) -> None:
        cfg = PipelineConfig(vlm_off=False, sample_id="DLPFC_151669")
        state = {"visual_ready": False}
        seen_envs = []
        seen_cmds = []

        repo = cfg.repo_root()
        scoring_script = repo / "scoring" / "scoring.py"
        pic_script = repo / "scoring" / "pic_analyze" / "run.py"
        visual_file = repo / "scoring" / "pic_analyze" / "output" / "all_domains_scores_DLPFC_151669.json"

        def _exists(path_obj):
            p = str(path_obj).replace("\\", "/")
            if p == str(scoring_script).replace("\\", "/"):
                return True
            if p == str(pic_script).replace("\\", "/"):
                return True
            if p == str(visual_file).replace("\\", "/"):
                return bool(state["visual_ready"])
            return True

        def _run_command(*, stage, env=None, cmd=None, **_kwargs):
            seen_envs.append(dict(env or {}))
            seen_cmds.append(list(cmd or []))
            if stage == "scoring_visual":
                state["visual_ready"] = True
            return {"returncode": 0, "interrupted": False, "stdout_tail": [], "log_line_count": 0}

        with patch.object(scoring_mod.Path, "exists", autospec=True, side_effect=_exists), patch.object(
            scoring_mod, "_run_command", side_effect=_run_command
        ):
            result = run_scoring(cfg)

        self.assertTrue(result["ok"])
        self.assertTrue(result["pic_analyze_autorun"])
        self.assertEqual(result["pic_analyze_exit_code"], 0)
        self.assertTrue(str(result["visual_scores_file"]).endswith("all_domains_scores_DLPFC_151669.json"))
        self.assertGreaterEqual(len(seen_envs), 2)
        for env in seen_envs:
            self.assertEqual(env.get("PYTHONIOENCODING"), "utf-8")
            self.assertEqual(env.get("PYTHONUTF8"), "1")
        visual_cmd = next(cmd for cmd in seen_cmds if any(str(part).endswith("run.py") for part in cmd))
        self.assertIn("--sample_id", visual_cmd)
        self.assertIn("DLPFC_151669", visual_cmd)

    def test_run_scoring_fails_when_pic_analyze_prestep_fails(self) -> None:
        cfg = PipelineConfig(vlm_off=False, sample_id="DLPFC_151669")

        repo = cfg.repo_root()
        scoring_script = repo / "scoring" / "scoring.py"
        pic_script = repo / "scoring" / "pic_analyze" / "run.py"
        visual_file = repo / "scoring" / "pic_analyze" / "output" / "all_domains_scores_DLPFC_151669.json"

        def _exists(path_obj):
            p = str(path_obj).replace("\\", "/")
            if p == str(scoring_script).replace("\\", "/"):
                return True
            if p == str(pic_script).replace("\\", "/"):
                return True
            if p == str(visual_file).replace("\\", "/"):
                return False
            return True

        with patch.object(scoring_mod.Path, "exists", autospec=True, side_effect=_exists), patch.object(
            scoring_mod,
            "_run_command",
            return_value={"returncode": 1, "interrupted": False, "stdout_tail": ["err"], "log_line_count": 1},
        ):
            result = run_scoring(cfg)

        self.assertFalse(result["ok"])
        self.assertTrue(result["pic_analyze_autorun"])
        self.assertIn("pic_analyze pre-step failed", result["error"])


if __name__ == "__main__":
    unittest.main()
