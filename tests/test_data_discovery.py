from __future__ import annotations

from contextlib import contextmanager
import shutil
import unittest
from pathlib import Path
import uuid

import yaml


def _make_visium_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    (base / "spatial").mkdir(parents=True, exist_ok=True)
    (base / "filtered_feature_bc_matrix.h5").write_text("", encoding="utf-8")
    return base


@contextmanager
def _workspace_case_dir():
    base = (Path.cwd() / "cache" / f"test_data_discovery_{uuid.uuid4().hex[:8]}").resolve()
    base.mkdir(parents=True, exist_ok=False)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


class DataDiscoveryTests(unittest.TestCase):
    def test_pipeline_config_has_highest_priority(self) -> None:
        from streamlit_app.utils.data_discovery import discover_data_defaults

        with _workspace_case_dir() as repo:
            visium_dir = _make_visium_dir(repo / "data" / "from_pipeline")
            csv_dir = (repo / "custom_scoring_input").resolve()
            csv_dir.mkdir(parents=True, exist_ok=True)
            (repo / "pipeline_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "data_path": str(visium_dir),
                        "sample_id": "PIPE_SAMPLE",
                        "csv_path": str(csv_dir),
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            result = discover_data_defaults(repo_root=repo)
            self.assertEqual(result["data_path"], str(visium_dir))
            self.assertEqual(result["sample_id"], "PIPE_SAMPLE")
            self.assertEqual(result["source"], "pipeline_config.yaml")
            self.assertEqual(result["csv_path"], str(csv_dir))
            self.assertEqual(result["csv_source"], "pipeline_config.yaml")

    def test_tool_runner_config_fallback_uses_config_sample_id(self) -> None:
        from streamlit_app.utils.data_discovery import discover_data_defaults

        with _workspace_case_dir() as repo:
            visium_dir = _make_visium_dir(repo / "Tool-runner" / "151507")
            cfg_dir = repo / "Tool-runner" / "configs"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "DLPFC_151507.yaml").write_text(
                yaml.safe_dump(
                    {
                        "data_path": str(visium_dir),
                        "sample_id": "DLPFC_151507",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            result = discover_data_defaults(repo_root=repo)
            self.assertEqual(result["data_path"], str(visium_dir))
            self.assertEqual(result["sample_id"], "DLPFC_151507")
            self.assertEqual(result["source"], "tool_runner_config")
            self.assertTrue(result["source_detail"].endswith("Tool-runner/configs/DLPFC_151507.yaml"))

    def test_tool_runner_config_relative_path_can_resolve_from_repo_root(self) -> None:
        from streamlit_app.utils.data_discovery import discover_data_defaults

        with _workspace_case_dir() as repo:
            _make_visium_dir(repo / "Tool-runner" / "151507")
            cfg_dir = repo / "Tool-runner" / "configs"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "DLPFC_151507.yaml").write_text(
                yaml.safe_dump(
                    {
                        "data_path": "Tool-runner/151507",
                        "sample_id": "DLPFC_151507",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            result = discover_data_defaults(repo_root=repo)
            self.assertTrue(result["data_path"].endswith("Tool-runner\\151507") or result["data_path"].endswith("Tool-runner/151507"))
            self.assertEqual(result["sample_id"], "DLPFC_151507")
            self.assertEqual(result["source"], "tool_runner_config")

    def test_repo_scan_fallback(self) -> None:
        from streamlit_app.utils.data_discovery import discover_data_defaults

        with _workspace_case_dir() as repo:
            visium_dir = _make_visium_dir(repo / "inputs" / "151507")

            result = discover_data_defaults(repo_root=repo)
            self.assertEqual(result["data_path"], str(visium_dir))
            self.assertEqual(result["sample_id"], "151507")
            self.assertEqual(result["source"], "repo_scan")
            self.assertEqual(result["csv_source"], "none")

    def test_default_scoring_input_is_second_csv_priority(self) -> None:
        from streamlit_app.utils.data_discovery import discover_data_defaults

        with _workspace_case_dir() as repo:
            _make_visium_dir(repo / "inputs" / "151507")
            default_csv = (repo / "scoring" / "input").resolve()
            default_csv.mkdir(parents=True, exist_ok=True)

            result = discover_data_defaults(repo_root=repo)
            self.assertEqual(result["csv_path"], str(default_csv))
            self.assertEqual(result["csv_source"], "scoring/input")

    def test_repo_scan_csv_fallback_when_default_missing(self) -> None:
        from streamlit_app.utils.data_discovery import discover_data_defaults

        with _workspace_case_dir() as repo:
            _make_visium_dir(repo / "inputs" / "151507")
            scanned_csv_dir = (repo / "generated" / "csvs").resolve()
            scanned_csv_dir.mkdir(parents=True, exist_ok=True)
            (scanned_csv_dir / "IRIS_DLPFC_151507_spot.csv").write_text("spot_id,x,y\n", encoding="utf-8")

            result = discover_data_defaults(repo_root=repo)
            self.assertEqual(result["csv_path"], str(scanned_csv_dir))
            self.assertEqual(result["csv_source"], "repo_scan")


if __name__ == "__main__":
    unittest.main()
