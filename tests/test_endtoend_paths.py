from pathlib import Path
import unittest


class EndToEndPathTests(unittest.TestCase):
    def test_endtoend_has_no_hardcoded_repo_name_segments(self) -> None:
        p = Path(__file__).resolve().parent.parent / "endtoend.py"
        text = p.read_text(encoding="utf-8")
        self.assertNotIn('repo_root / "EnsAgent"', text)


if __name__ == "__main__":
    unittest.main()
