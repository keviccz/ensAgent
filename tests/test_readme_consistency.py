from __future__ import annotations

import unittest
from pathlib import Path


class ReadmeConsistencyTests(unittest.TestCase):
    def test_readme_does_not_reference_missing_environment_fast_file(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        text = (repo / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("environment.fast.yml", text)

    def test_readme_uses_chevron_history_menu_wording(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        text = (repo / "README.md").read_text(encoding="utf-8")
        self.assertIn("right-side chevron menu", text)
        self.assertNotIn("use the `...` menu on each history item", text)


if __name__ == "__main__":
    unittest.main()
