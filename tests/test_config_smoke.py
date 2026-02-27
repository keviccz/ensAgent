import importlib
import unittest


class ConfigSmokeTests(unittest.TestCase):
    def test_scoring_config_imports(self) -> None:
        mod = importlib.import_module("scoring.config")
        self.assertTrue(hasattr(mod, "load_config"))


if __name__ == "__main__":
    unittest.main()
