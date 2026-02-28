from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path


def _load_endtoend_module():
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "endtoend.py"
    spec = importlib.util.spec_from_file_location("endtoend_module_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load endtoend module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    yaml_stub = types.SimpleNamespace(safe_load=lambda _text: {}, dump=lambda *_args, **_kwargs: "")
    import sys
    old_yaml = sys.modules.get("yaml")
    try:
        sys.modules["yaml"] = yaml_stub
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        if old_yaml is not None:
            sys.modules["yaml"] = old_yaml
        else:
            sys.modules.pop("yaml", None)
    return module


class _Cfg:
    def __init__(
        self,
        *,
        skip_tool_runner: bool,
        skip_scoring: bool,
        run_best: bool,
        run_annotation_multiagent: bool,
        sample_id: str = "",
        data_path: str = "",
    ):
        self.skip_tool_runner = skip_tool_runner
        self.skip_scoring = skip_scoring
        self.run_best = run_best
        self.run_annotation_multiagent = run_annotation_multiagent
        self.sample_id = sample_id
        self.data_path = data_path


class EndToEndCliBehaviorTests(unittest.TestCase):
    def test_no_run_best_overrides_to_false(self) -> None:
        mod = _load_endtoend_module()
        parser = mod._build_parser()
        args = parser.parse_args(["--no_run_best"])
        overrides = mod._build_cli_overrides(args)
        self.assertIn("run_best", overrides)
        self.assertFalse(overrides["run_best"])

    def test_no_run_annotation_overrides_to_false(self) -> None:
        mod = _load_endtoend_module()
        parser = mod._build_parser()
        args = parser.parse_args(["--no_run_annotation_multiagent"])
        overrides = mod._build_cli_overrides(args)
        self.assertIn("run_annotation_multiagent", overrides)
        self.assertFalse(overrides["run_annotation_multiagent"])

    def test_validate_allows_annotation_without_data_path(self) -> None:
        mod = _load_endtoend_module()
        cfg = _Cfg(
            skip_tool_runner=True,
            skip_scoring=True,
            run_best=False,
            run_annotation_multiagent=True,
            sample_id="DLPFC_151507",
            data_path="",
        )
        self.assertEqual(mod._validate_required_fields(cfg), [])

    def test_validate_requires_data_path_when_tool_runner_enabled(self) -> None:
        mod = _load_endtoend_module()
        cfg = _Cfg(
            skip_tool_runner=False,
            skip_scoring=True,
            run_best=False,
            run_annotation_multiagent=False,
            sample_id="DLPFC_151507",
            data_path="",
        )
        errs = mod._validate_required_fields(cfg)
        self.assertTrue(any("data_path is required" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
