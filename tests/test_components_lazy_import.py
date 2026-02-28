from __future__ import annotations

import importlib
import sys
import unittest


class ComponentsLazyImportTests(unittest.TestCase):
    def test_package_import_does_not_eager_import_submodules(self) -> None:
        # Reset component package/module cache for deterministic assertion.
        for mod_name in list(sys.modules):
            if mod_name == "streamlit_app.components" or mod_name.startswith("streamlit_app.components."):
                sys.modules.pop(mod_name, None)

        pkg = importlib.import_module("streamlit_app.components")
        self.assertIsNotNone(pkg)
        self.assertNotIn("streamlit_app.components.chat", sys.modules)
        self.assertNotIn("streamlit_app.components.sidebar", sys.modules)
        self.assertNotIn("streamlit_app.components.settings", sys.modules)


if __name__ == "__main__":
    unittest.main()
