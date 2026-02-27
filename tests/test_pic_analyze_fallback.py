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
    spec = importlib.util.spec_from_file_location("pic_analyze_auto_analyzer_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyImageManager:
    def list_images(self):
        return [
            {"filename": "GraphST_domain.png"},
            {"filename": "IRIS_domain.png"},
        ]


class PicAnalyzeFallbackTests(unittest.TestCase):
    def test_ocr_capability_error_auto_fallback_stops_retry_loop(self) -> None:
        mod = _load_auto_analyzer_module()
        analyzer = mod.AutoClusteringAnalyzer.__new__(mod.AutoClusteringAnalyzer)
        analyzer.image_manager = _DummyImageManager()
        analyzer.all_scores = {}

        def _raise_ocr_error(_domain: int):
            raise RuntimeError("Configured model does not support OCR/vision image input.")

        analyzer.analyze_domain = _raise_ocr_error

        with patch.object(mod.time, "sleep") as mock_sleep, patch("builtins.print"):
            ok = analyzer.run_all_domains_analysis(domains=[1])

        self.assertFalse(ok)
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
