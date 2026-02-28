from pathlib import Path
import unittest


class MainPageHeaderTests(unittest.TestCase):
    def test_settings_page_uses_single_header_source(self) -> None:
        p = Path(__file__).resolve().parent.parent / "streamlit_app" / "main.py"
        text = p.read_text(encoding="utf-8")
        self.assertIn('if active_page not in {"chat", "settings"}:', text)


if __name__ == "__main__":
    unittest.main()
