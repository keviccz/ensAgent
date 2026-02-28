from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_pipeline_module():
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "ensagent_tools" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("ensagent_pipeline_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pipeline module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    yaml_stub = types.SimpleNamespace(safe_load=lambda _text: {}, dump=lambda *_args, **_kwargs: "")
    with patch.dict("sys.modules", {"yaml": yaml_stub}):
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class _DummyCfg:
    def __init__(self, *, repo_root: Path, skip_scoring: bool, run_best: bool, run_annotation: bool):
        self._repo_root = repo_root
        self.skip_tool_runner = True
        self.skip_scoring = skip_scoring
        self.run_best = run_best
        self.run_annotation_multiagent = run_annotation
        self.overwrite_staging = False
        self.sample_id = "DLPFC_151507"

    def repo_root(self) -> Path:
        return self._repo_root

    def resolved_tool_output_dir(self) -> Path:
        return self._repo_root / "output" / "tool_runner" / self.sample_id

    def resolved_best_output_dir(self) -> Path:
        return self._repo_root / "output" / "best" / self.sample_id


class PipelineStageIndependenceTests(unittest.TestCase):
    def test_skip_scoring_does_not_trigger_staging(self) -> None:
        mod = _load_pipeline_module()
        cfg = _DummyCfg(
            repo_root=Path(__file__).resolve().parent.parent,
            skip_scoring=True,
            run_best=True,
            run_annotation=True,
        )

        with patch.object(mod, "stage_toolrunner_outputs") as mock_stage, patch.object(
            mod, "run_best_builder", return_value={"ok": True}
        ) as mock_best, patch.object(mod, "run_annotation", return_value={"ok": True}) as mock_ann:
            result = mod.run_full_pipeline(cfg)

        self.assertTrue(result["ok"])
        mock_stage.assert_not_called()
        mock_best.assert_called_once()
        mock_ann.assert_called_once()
        self.assertTrue(result["phases"]["staging"]["skipped"])

    def test_scoring_enabled_triggers_staging(self) -> None:
        mod = _load_pipeline_module()
        cfg = _DummyCfg(
            repo_root=Path(__file__).resolve().parent.parent,
            skip_scoring=False,
            run_best=False,
            run_annotation=False,
        )

        with patch.object(mod, "stage_toolrunner_outputs") as mock_stage, patch.object(
            mod, "run_scoring", return_value={"ok": True}
        ) as mock_scoring:
            result = mod.run_full_pipeline(cfg)

        self.assertTrue(result["ok"])
        mock_stage.assert_called_once()
        mock_scoring.assert_called_once()


if __name__ == "__main__":
    unittest.main()
