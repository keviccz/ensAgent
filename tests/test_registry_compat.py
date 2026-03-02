from __future__ import annotations

import unittest
from unittest.mock import patch

from ensagent_tools.registry import execute_tool


class RegistryCompatTests(unittest.TestCase):
    def test_execute_tool_retries_without_progress_kwargs_for_legacy_signature(self) -> None:
        calls = {"n": 0}

        def legacy_run_scoring(_cfg, *, input_dir: str = ""):
            calls["n"] += 1
            return {"ok": True, "input_dir": input_dir}

        with patch.dict("ensagent_tools.registry.TOOL_REGISTRY", {"run_scoring": legacy_run_scoring}, clear=False):
            out = execute_tool(
                "run_scoring",
                {"input_dir": "scoring/input"},
                object(),
                progress_callback=lambda _evt: None,
                cancel_check=lambda: False,
            )

        self.assertTrue(out["ok"])
        self.assertEqual(out["input_dir"], "scoring/input")
        self.assertTrue(out.get("compat_retry_used"))
        self.assertEqual(calls["n"], 1)

    def test_execute_tool_uses_new_signature_without_retry(self) -> None:
        seen = {}

        def modern_run_scoring(_cfg, *, input_dir: str = "", progress_callback=None, cancel_check=None):
            seen["input_dir"] = input_dir
            seen["has_progress"] = callable(progress_callback)
            seen["has_cancel"] = callable(cancel_check)
            return {"ok": True}

        with patch.dict("ensagent_tools.registry.TOOL_REGISTRY", {"run_scoring": modern_run_scoring}, clear=False):
            out = execute_tool(
                "run_scoring",
                {"input_dir": "scoring/input"},
                object(),
                progress_callback=lambda _evt: None,
                cancel_check=lambda: False,
            )

        self.assertTrue(out["ok"])
        self.assertEqual(seen["input_dir"], "scoring/input")
        self.assertTrue(seen["has_progress"])
        self.assertTrue(seen["has_cancel"])
        self.assertFalse(out.get("compat_retry_used", False))

    def test_execute_tool_does_not_retry_unrelated_type_error(self) -> None:
        def bad_tool(_cfg, **_kw):
            raise TypeError("bad payload shape")

        with patch.dict("ensagent_tools.registry.TOOL_REGISTRY", {"run_scoring": bad_tool}, clear=False):
            out = execute_tool("run_scoring", {"input_dir": "x"}, object(), progress_callback=lambda _evt: None)

        self.assertFalse(out["ok"])
        self.assertIn("bad payload shape", out["error"])

    def test_execute_tool_skips_none_progress_injection(self) -> None:
        seen = {}

        def inspect_kwargs(_cfg, **kwargs):
            seen.update(kwargs)
            return {"ok": True}

        with patch.dict("ensagent_tools.registry.TOOL_REGISTRY", {"run_scoring": inspect_kwargs}, clear=False):
            out = execute_tool("run_scoring", {"input_dir": "x"}, object())

        self.assertTrue(out["ok"])
        self.assertEqual(seen.get("input_dir"), "x")
        self.assertNotIn("progress_callback", seen)
        self.assertNotIn("cancel_check", seen)


if __name__ == "__main__":
    unittest.main()
