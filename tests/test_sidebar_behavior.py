import unittest
from unittest.mock import patch

from streamlit_app.components import sidebar as sidebar_mod


class SidebarBehaviorTests(unittest.TestCase):
    def test_open_conversation_from_history_switches_to_chat(self) -> None:
        with patch.object(sidebar_mod, "set_state") as mock_set_state, patch.object(
            sidebar_mod, "load_conversation"
        ) as mock_load:
            sidebar_mod._open_conversation_from_history("abc123")

        mock_set_state.assert_called_once_with("active_page", "chat")
        mock_load.assert_called_once_with("abc123")


if __name__ == "__main__":
    unittest.main()
