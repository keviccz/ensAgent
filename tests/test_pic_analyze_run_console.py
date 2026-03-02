from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_run_module():
    repo = Path(__file__).resolve().parent.parent
    module_path = repo / "scoring" / "pic_analyze" / "run.py"
    spec = importlib.util.spec_from_file_location("pic_analyze_run_console_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeGbkFile:
    encoding = "gbk"

    def __init__(self) -> None:
        self.content = ""
        self.flushed = False

    def write(self, text: str) -> int:
        self.content += text
        return len(text)

    def flush(self) -> None:
        self.flushed = True


class PicAnalyzeRunConsoleTests(unittest.TestCase):
    def test_safe_print_handles_unicode_encode_error(self) -> None:
        mod = _load_run_module()
        fake_file = _FakeGbkFile()

        def _raise_unicode_error(*_args, **_kwargs):
            raise UnicodeEncodeError("gbk", "🚀", 0, 1, "illegal multibyte sequence")

        with patch.object(mod, "_ORIGINAL_PRINT", side_effect=_raise_unicode_error):
            mod._safe_print("🚀 启动", file=fake_file, end="", flush=True)

        self.assertTrue(fake_file.flushed)
        self.assertNotIn("🚀", fake_file.content)
        self.assertIn("?", fake_file.content)


if __name__ == "__main__":
    unittest.main()
