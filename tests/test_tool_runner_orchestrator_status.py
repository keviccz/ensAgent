from __future__ import annotations

import importlib.util
import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_tool_runner_agent_class():
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "Tool-runner" / "orchestrator.py"
    spec = importlib.util.spec_from_file_location("tool_runner_orchestrator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load orchestrator module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    yaml_stub = types.SimpleNamespace(safe_load=lambda _text: {})
    with patch.dict("sys.modules", {"yaml": yaml_stub}):
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.ToolRunnerAgent


class ToolRunnerStatusTests(unittest.TestCase):
    def _make_agent(self):
        ToolRunnerAgent = _load_tool_runner_agent_class()
        repo_root = Path(__file__).resolve().parent.parent
        data_dir = repo_root
        out_dir = repo_root / "tests"
        report_path = out_dir / "tool_runner_report.json"
        self.addCleanup(lambda: report_path.unlink(missing_ok=True))

        agent = ToolRunnerAgent(
            data_path=str(data_dir),
            sample_id="DLPFC_151507",
            output_dir=str(out_dir),
            methods=[],
        )
        return agent, out_dir

    def test_generate_report_preserves_failed_status(self) -> None:
        agent, out_dir = self._make_agent()
        agent.results["status"] = "failed"
        agent.results["error"] = "forced failure"
        agent.generate_report()

        report_path = out_dir / "tool_runner_report.json"
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "failed")

    def test_run_marks_partial_success_when_downstream_fails(self) -> None:
        agent, out_dir = self._make_agent()
        agent.run_clustering_tools = lambda: True
        agent.run_alignment = lambda: True
        agent.run_downstream_analysis = lambda: {
            "all_ok": False,
            "succeeded_analyses": ["DEGs", "Spots"],
            "failed_analyses": ["Pathways"],
        }

        with patch("builtins.print"):
            ok = agent.run()
        self.assertTrue(ok)
        self.assertEqual(agent.results["status"], "partial_success")
        self.assertIn("Pathways", agent.results["failed_analyses"])
        self.assertTrue(agent.results.get("warnings"))

        report_path = out_dir / "tool_runner_report.json"
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "partial_success")
        self.assertIn("Pathways", payload.get("failed_analyses", []))

    def test_run_returns_false_on_alignment_failure(self) -> None:
        agent, out_dir = self._make_agent()
        agent.run_clustering_tools = lambda: True
        agent.run_alignment = lambda: False

        with patch("builtins.print"):
            ok = agent.run()
        self.assertFalse(ok)
        self.assertEqual(agent.results["status"], "failed")

        report_path = out_dir / "tool_runner_report.json"
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "failed")


if __name__ == "__main__":
    unittest.main()
