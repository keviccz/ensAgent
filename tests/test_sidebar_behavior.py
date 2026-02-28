import unittest
from unittest.mock import patch
from pathlib import Path

from streamlit_app.components import sidebar as sidebar_mod


class SidebarBehaviorTests(unittest.TestCase):
    def test_open_conversation_from_history_switches_to_chat(self) -> None:
        with patch.object(sidebar_mod, "set_state") as mock_set_state, patch.object(
            sidebar_mod, "load_conversation"
        ) as mock_load:
            sidebar_mod._open_conversation_from_history("abc123")

        mock_set_state.assert_called_once_with("active_page", "chat")
        mock_load.assert_called_once_with("abc123")

    def test_history_menu_popover_does_not_use_manual_arrow_label(self) -> None:
        p = Path(__file__).resolve().parent.parent / "streamlit_app" / "components" / "sidebar.py"
        text = p.read_text(encoding="utf-8")
        self.assertNotIn('st.popover("▾"', text)
        self.assertNotIn('st.popover("⌄"', text)
        self.assertIn('st.popover("", use_container_width=True)', text)


if __name__ == "__main__":
    unittest.main()
