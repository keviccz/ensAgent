from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_auto_analyzer_module():
    repo = Path(__file__).resolve().parent.parent
    module_path = repo / "scoring" / "pic_analyze" / "auto_analyzer.py"
    module_dir = module_path.parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    spec = importlib.util.spec_from_file_location("pic_analyze_auto_analyzer_sample_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyImageManager:
    def __init__(self):
        pass


class PicAnalyzeSampleContextTests(unittest.TestCase):
    def test_auto_analyzer_uses_sample_specific_prompt_and_output_names(self) -> None:
        mod = _load_auto_analyzer_module()

        with patch.object(mod, "AzureOpenAIClient", side_effect=RuntimeError("missing runtime")), patch.object(
            mod, "_load_image_manager_class", return_value=_DummyImageManager
        ), patch.object(mod.os, "makedirs"), patch.object(mod.AutoClusteringAnalyzer, "_load_existing_scores"):
            analyzer = mod.AutoClusteringAnalyzer(sample_id="DLPFC_151669")

        self.assertEqual(analyzer.sample_id, "DLPFC_151669")
        self.assertIn("DLPFC_151669", analyzer.domain_prompts[1])
        self.assertNotIn("(151507)", analyzer.domain_prompts[1])
        self.assertEqual(analyzer.summary_json_name, "all_domains_scores_DLPFC_151669.json")
        self.assertEqual(analyzer.report_filename_for_domain(3), "domain3_report_DLPFC_151669.txt")


if __name__ == "__main__":
    unittest.main()
