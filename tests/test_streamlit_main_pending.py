import unittest
from unittest.mock import patch

from streamlit_app import main as main_mod


class StreamlitMainPendingTests(unittest.TestCase):
    def test_main_processes_pending_response_before_render(self) -> None:
        order: list[str] = []

        with patch.object(main_mod, "init_session_state", side_effect=lambda: order.append("init")), patch.object(
            main_mod, "process_pending_response", side_effect=lambda: order.append("process")
        ), patch.object(main_mod.st, "markdown", side_effect=lambda *_a, **_k: order.append("markdown")), patch.object(
            main_mod, "get_premium_css", return_value=""
        ), patch.object(main_mod, "render_sidebar", side_effect=lambda: order.append("sidebar")), patch.object(
            main_mod, "render_main_content", side_effect=lambda: order.append("content")
        ):
            main_mod.main()

        self.assertGreaterEqual(len(order), 5)
        self.assertEqual(order[:3], ["init", "process", "markdown"])


if __name__ == "__main__":
    unittest.main()
