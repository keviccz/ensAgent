from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_best_builder_module():
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "ensagent_tools" / "best_builder.py"
    spec = importlib.util.spec_from_file_location("ensagent_best_builder_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load best_builder module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    yaml_stub = types.SimpleNamespace(safe_load=lambda _text: {}, dump=lambda *_args, **_kwargs: "")
    with patch.dict("sys.modules", {"yaml": yaml_stub}):
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class _Cfg:
    def __init__(self, repo_root: Path, *, sample_id: str, csv_dir: Path):
        self._repo_root = repo_root
        self.sample_id = sample_id
        self.best_smooth_knn = False
        self.best_truth_file = ""
        self.data_path = ""
        self._csv_dir = csv_dir

    def repo_root(self) -> Path:
        return self._repo_root

    def resolved_best_output_dir(self) -> Path:
        return self._repo_root / "output" / "best" / self.sample_id

    def resolved_scoring_input_dir(self) -> Path:
        return self._csv_dir


class BestBuilderPathTests(unittest.TestCase):
    def test_best_builder_prefers_cfg_csv_path_for_spot_template(self) -> None:
        mod = _load_best_builder_module()
        sid = "DLPFC_151507"
        root = Path("repo")
        cfg_csv = Path("cfg_input")
        default_csv = root / "scoring" / "input"
        cfg_template = cfg_csv / f"IRIS_{sid}_spot.csv"
        default_template = default_csv / f"SEDR_{sid}_spot.csv"
        scores_matrix = root / "scoring" / "output" / "consensus" / "scores_matrix.csv"
        labels_matrix = root / "scoring" / "output" / "consensus" / "labels_matrix.csv"
        script_path = root / "ensemble" / "build_best.py"
        cfg = _Cfg(root, sample_id=sid, csv_dir=cfg_csv)

        def _exists(path_obj: Path) -> bool:
            normalized = str(path_obj).replace("\\", "/")
            if normalized in {
                str(script_path).replace("\\", "/"),
                str(scores_matrix).replace("\\", "/"),
                str(labels_matrix).replace("\\", "/"),
                str(cfg_template).replace("\\", "/"),
                str(default_template).replace("\\", "/"),
                str(cfg_csv).replace("\\", "/"),
                str(default_csv).replace("\\", "/"),
            }:
                return True
            return False

        def _is_dir(path_obj: Path) -> bool:
            normalized = str(path_obj).replace("\\", "/")
            return normalized in {
                str(cfg_csv).replace("\\", "/"),
                str(default_csv).replace("\\", "/"),
            }

        def _glob(path_obj: Path, pattern: str):
            if path_obj == cfg_csv and pattern == f"*{sid}*_spot.csv":
                return [cfg_template]
            if path_obj == default_csv and pattern == f"*{sid}*_spot.csv":
                return [default_template]
            return []

        with patch.object(mod.Path, "exists", autospec=True, side_effect=_exists), patch.object(
            mod.Path, "is_dir", autospec=True, side_effect=_is_dir
        ), patch.object(mod.Path, "glob", autospec=True, side_effect=_glob), patch.object(
            mod.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)
        ):
            res = mod.run_best_builder(cfg)

        self.assertTrue(res["ok"])
        self.assertEqual(res["spot_template_used"], str(cfg_template))

    def test_best_builder_reports_missing_scores_matrix(self) -> None:
        mod = _load_best_builder_module()
        sid = "DLPFC_151507"
        root = Path("repo")
        cfg_csv = Path("cfg_input")
        cfg = _Cfg(root, sample_id=sid, csv_dir=cfg_csv)
        script_path = root / "ensemble" / "build_best.py"
        labels_matrix = root / "scoring" / "output" / "consensus" / "labels_matrix.csv"

        def _exists(path_obj: Path) -> bool:
            normalized = str(path_obj).replace("\\", "/")
            if normalized in {
                str(script_path).replace("\\", "/"),
                str(labels_matrix).replace("\\", "/"),
            }:
                return True
            return False

        with patch.object(mod.Path, "exists", autospec=True, side_effect=_exists):
            res = mod.run_best_builder(cfg)

        self.assertFalse(res["ok"])
        self.assertIn("scores_matrix not found", res["error"])


if __name__ == "__main__":
    unittest.main()
