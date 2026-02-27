import unittest

from streamlit_app.styles import get_premium_css


class SidebarStylesTests(unittest.TestCase):
    def test_sidebar_collapse_controls_are_forced_visible(self) -> None:
        css = get_premium_css()
        self.assertIn('[data-testid="stSidebarCollapseButton"]', css)
        self.assertIn('button[aria-label="Collapse sidebar"]', css)
        self.assertIn("opacity: 1 !important;", css)

    def test_new_chat_button_color_is_black(self) -> None:
        css = get_premium_css()
        self.assertIn(".ens-new-chat-anchor", css)
        self.assertIn("color: #000000 !important;", css)

    def test_recent_history_titles_use_compact_font_size(self) -> None:
        css = get_premium_css()
        self.assertIn(".ens-history-list-anchor", css)
        self.assertIn("font-size: 0.62rem !important;", css)


if __name__ == "__main__":
    unittest.main()
