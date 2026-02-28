from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ensagent_tools.config_manager import PipelineConfig
from ensagent_tools import env_manager as env_mod


class EnvManagerResolverTests(unittest.TestCase):
    def test_resolve_prefers_existing_configured_executable_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exe_path = Path(tmp) / "mamba.exe"
            exe_path.write_text("", encoding="utf-8")
            cfg = PipelineConfig(conda_exe=str(exe_path))

            resolved = env_mod.resolve_conda_executable(cfg)

        self.assertTrue(resolved["ok"])
        self.assertEqual(resolved["exe"], str(exe_path))
        self.assertEqual(resolved["source"], "config")

    def test_resolve_falls_back_to_path_when_configured_missing(self) -> None:
        cfg = PipelineConfig(conda_exe="missing_conda_cmd")

        def _which(name: str) -> str | None:
            if name == "mamba":
                return os.path.join("C:\\", "Tools", "mamba.exe")
            return None

        with patch.object(env_mod.shutil, "which", side_effect=_which):
            resolved = env_mod.resolve_conda_executable(cfg)

        self.assertTrue(resolved["ok"])
        self.assertEqual(resolved["source"], "path")
        self.assertTrue(
            any(item["candidate"] == "mamba" and item["resolved"] for item in resolved["checked"])
        )

    def test_check_envs_reports_missing_executable_with_diagnostics(self) -> None:
        cfg = PipelineConfig(conda_exe="definitely_missing")
        with patch.object(env_mod, "resolve_conda_executable", return_value={
            "ok": False,
            "exe": "",
            "source": "",
            "requested": "definitely_missing",
            "checked": [{"candidate": "definitely_missing", "source": "config", "resolved": ""}],
        }):
            out = env_mod.check_envs(cfg)

        self.assertFalse(out["ok"])
        self.assertIn("resolver_checked", out)
        self.assertTrue(out["warnings"])


if __name__ == "__main__":
    unittest.main()
