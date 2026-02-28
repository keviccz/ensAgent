from __future__ import annotations

import importlib.util
import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class PicAnalyzeImportResilienceTests(unittest.TestCase):
    def test_missing_optional_deps_do_not_crash_import(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        pic_dir = repo / "scoring" / "pic_analyze"

        fake_dotenv = types.ModuleType("dotenv")  # no load_dotenv attr -> ImportError in from-import
        fake_openai = types.ModuleType("openai")  # no AzureOpenAI attr -> ImportError in from-import
        fake_yaml = types.SimpleNamespace(safe_load=lambda _text: {})

        image_manager_stub = types.ModuleType("image_manager")

        class _StubImageManager:
            def __init__(self):
                pass

        image_manager_stub.ImageManager = _StubImageManager

        with patch.dict(
            sys.modules,
            {
                "dotenv": fake_dotenv,
                "openai": fake_openai,
                "yaml": fake_yaml,
                "image_manager": image_manager_stub,
            },
        ):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")

                config_mod = _load_module("pic_config_test", pic_dir / "config.py")
                # azure_client fallback path imports "config"
                sys.modules["config"] = config_mod
                azure_mod = _load_module("pic_azure_client_test", pic_dir / "azure_client.py")
                # auto_analyzer fallback path imports "azure_client"
                sys.modules["azure_client"] = azure_mod
                analyzer_mod = _load_module("pic_auto_analyzer_test", pic_dir / "auto_analyzer.py")

                analyzer = analyzer_mod.AutoClusteringAnalyzer()
                sys.modules.pop("config", None)
                sys.modules.pop("azure_client", None)

        self.assertIsNotNone(config_mod.Config)
        self.assertIsNotNone(azure_mod.AzureOpenAIClient)
        self.assertIsNone(analyzer.azure_client)
        joined = "\n".join(str(w.message) for w in captured)
        self.assertIn("python-dotenv", joined)
        self.assertIn("openai package", joined)


if __name__ == "__main__":
    unittest.main()
