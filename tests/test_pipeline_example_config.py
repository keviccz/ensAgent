from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class PipelineExampleConfigTests(unittest.TestCase):
    def test_example_contains_repo_local_151507_paths(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        cfg_path = repo / "pipeline_config.example.yaml"
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

        self.assertEqual(data.get("data_path"), "Tool-runner/151507")
        self.assertEqual(data.get("sample_id"), "DLPFC_151507")
        self.assertEqual(data.get("tool_output_dir"), "output/tool_runner/DLPFC_151507")
        self.assertEqual(data.get("best_output_dir"), "output/best/DLPFC_151507")
        self.assertEqual(data.get("best_truth_file"), "Tool-runner/151507/151507_truth.txt")
        self.assertIn("vlm_off", data)


if __name__ == "__main__":
    unittest.main()
