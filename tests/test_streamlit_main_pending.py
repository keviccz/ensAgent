import unittest
from unittest.mock import patch

from streamlit_app import main as main_mod


class StreamlitMainPendingTests(unittest.TestCase):
    def test_main_processes_pending_response_before_render(self) -> None:
        order: list[str] = []

        with patch.object(main_mod, "init_session_state", side_effect=lambda: order.append("init")), patch.object(
            main_mod, "initialize_sidebar_state", side_effect=lambda: order.append("sidebar_init")
        ), patch.object(
            main_mod, "process_pending_response", side_effect=lambda: order.append("process")
        ), patch.object(
            main_mod, "consume_pending_action_prompt", side_effect=lambda: order.append("consume_action")
        ), patch.object(main_mod.st, "markdown", side_effect=lambda *_a, **_k: order.append("markdown")), patch.object(
            main_mod, "get_premium_css", return_value=""
        ), patch.object(main_mod, "render_sidebar", side_effect=lambda: order.append("sidebar")), patch.object(
            main_mod, "render_main_content", side_effect=lambda: order.append("content")
        ), patch.object(
            main_mod, "_poll_pending_response", side_effect=lambda: order.append("poll")
        ):
            main_mod.main()

        self.assertGreaterEqual(len(order), 8)
        self.assertEqual(order[:5], ["init", "sidebar_init", "process", "consume_action", "markdown"])

    def test_poll_pending_response_reruns_when_pending(self) -> None:
        with patch.object(main_mod, "_has_unrendered_completion", return_value=False), patch.object(
            main_mod, "_has_pending_response", return_value=True
        ), patch.object(
            main_mod.time, "sleep"
        ) as mock_sleep, patch.object(main_mod.st, "rerun") as mock_rerun:
            main_mod._poll_pending_response()

        mock_sleep.assert_called_once_with(main_mod._PENDING_POLL_INTERVAL_SEC)
        mock_rerun.assert_called_once()

    def test_poll_pending_response_noop_without_pending(self) -> None:
        with patch.object(main_mod, "_has_unrendered_completion", return_value=False), patch.object(
            main_mod, "_has_pending_response", return_value=False
        ), patch.object(
            main_mod.time, "sleep"
        ) as mock_sleep, patch.object(main_mod.st, "rerun") as mock_rerun:
            main_mod._poll_pending_response()

        mock_sleep.assert_not_called()
        mock_rerun.assert_not_called()

    def test_poll_pending_response_reruns_on_unrendered_completion(self) -> None:
        with patch.object(main_mod, "_has_unrendered_completion", return_value=True), patch.object(
            main_mod, "get_state", return_value=3
        ), patch.object(main_mod, "set_state") as mock_set_state, patch.object(
            main_mod.st, "rerun"
        ) as mock_rerun, patch.object(main_mod.time, "sleep") as mock_sleep:
            main_mod._poll_pending_response()

        mock_set_state.assert_called_once_with("_chat_completion_seen_tick", 3)
        mock_rerun.assert_called_once()
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
