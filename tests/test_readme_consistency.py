from __future__ import annotations

import unittest
from pathlib import Path


class ReadmeConsistencyTests(unittest.TestCase):
    def test_readme_does_not_reference_missing_environment_fast_file(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        text = (repo / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("environment.fast.yml", text)

    def test_readme_documents_nextjs_launcher_and_not_streamlit(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        text = (repo / "README.md").read_text(encoding="utf-8")
        self.assertIn("python start.py", text)
        self.assertNotIn("streamlit run streamlit_app/main.py", text)
        self.assertNotIn("## Streamlit UI", text)


if __name__ == "__main__":
    unittest.main()
