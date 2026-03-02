import unittest

from streamlit_app.styles import get_premium_css


class SidebarStylesTests(unittest.TestCase):
    def test_sidebar_collapse_controls_are_forced_visible(self) -> None:
        css = get_premium_css()
        self.assertIn('[data-testid="stSidebarCollapseButton"]', css)
        self.assertIn('button[aria-label="Collapse sidebar"]', css)
        self.assertIn("opacity: 1 !important;", css)

    def test_new_chat_button_color_uses_theme_variable(self) -> None:
        css = get_premium_css()
        self.assertIn(".ens-new-chat-anchor", css)
        self.assertIn("color: var(--new-chat-color) !important;", css)

    def test_recent_history_titles_use_compact_font_size(self) -> None:
        css = get_premium_css()
        self.assertIn(".ens-history-list-anchor", css)
        self.assertIn("font-size: 0.44rem !important;", css)

    def test_recent_label_font_size_is_compact(self) -> None:
        css = get_premium_css()
        self.assertIn(".ens-history-label", css)
        self.assertIn("font-size: 0.52rem;", css)

    def test_sidebar_buttons_are_square(self) -> None:
        css = get_premium_css()
        self.assertIn('section[data-testid="stSidebar"] .stButton > button', css)
        self.assertIn("border-radius: 0 !important;", css)

    def test_history_popover_trigger_is_borderless(self) -> None:
        css = get_premium_css()
        self.assertIn('[data-testid="stPopover"] > button', css)
        self.assertIn('[data-testid="stPopover"] > *', css)
        self.assertIn('[data-testid="stPopover"] [role="button"]', css)
        self.assertIn('[data-testid="stPopover"] [data-baseweb="button"]', css)
        self.assertIn("border: 0 !important;", css)
        self.assertIn("background: transparent !important;", css)
        self.assertIn("color: var(--history-menu-ink) !important;", css)
        self.assertIn('[aria-expanded="true"]', css)
        self.assertIn("font-size: 0 !important;", css)
        self.assertIn("content: none !important;", css)

    def test_dark_theme_overrides_are_removed(self) -> None:
        css = get_premium_css()
        self.assertNotIn('[data-theme="dark"]', css)
        self.assertNotIn("prefers-color-scheme: dark", css)
        self.assertNotIn(".ens-theme-toggle-anchor", css)

    def test_history_popover_actions_share_same_button_style(self) -> None:
        css = get_premium_css()
        self.assertIn('[data-testid="stPopoverPopover"] .stButton > button', css)
        self.assertIn('[data-testid="stPopoverPopover"] .stDownloadButton > button', css)
        self.assertIn("border-radius: 0 !important;", css)
        self.assertIn("background: var(--surface) !important;", css)


if __name__ == "__main__":
    unittest.main()
