import types
import unittest
from unittest.mock import patch

from streamlit_app.components import settings as settings_mod


class SettingsRateLimitTests(unittest.TestCase):
    def test_rejects_too_fast_numeric_updates(self) -> None:
        state = {"temperature": 0.7}
        fake_st = types.SimpleNamespace(session_state={"_temp_input": 1.2})

        def _get_state(key, default=None):
            return state.get(key, default)

        def _set_state(key, value):
            state[key] = value

        with patch.object(settings_mod, "st", fake_st), patch.object(
            settings_mod, "get_state", side_effect=_get_state
        ), patch.object(settings_mod, "set_state", side_effect=_set_state), patch.object(
            settings_mod.time, "monotonic", side_effect=[10.0, 10.05]
        ):
            self.assertTrue(settings_mod._allow_numeric_update("temperature", "_temp_input"))
            fake_st.session_state["_temp_input"] = 1.3
            self.assertFalse(settings_mod._allow_numeric_update("temperature", "_temp_input"))

        self.assertEqual(fake_st.session_state["_temp_input"], 0.7)
        self.assertIn("_settings_rate_notice", state)

    def test_rejects_over_quota_updates_within_window(self) -> None:
        state = {"top_p": 1.0}
        fake_st = types.SimpleNamespace(session_state={"_top_p_input": 0.95})

        def _get_state(key, default=None):
            return state.get(key, default)

        def _set_state(key, value):
            state[key] = value

        with patch.object(settings_mod, "st", fake_st), patch.object(
            settings_mod, "get_state", side_effect=_get_state
        ), patch.object(settings_mod, "set_state", side_effect=_set_state), patch.object(
            settings_mod.time, "monotonic", side_effect=[1.0, 1.2, 1.4, 1.6, 1.8]
        ):
            self.assertTrue(settings_mod._allow_numeric_update("top_p", "_top_p_input"))
            self.assertTrue(settings_mod._allow_numeric_update("top_p", "_top_p_input"))
            self.assertTrue(settings_mod._allow_numeric_update("top_p", "_top_p_input"))
            self.assertTrue(settings_mod._allow_numeric_update("top_p", "_top_p_input"))
            fake_st.session_state["_top_p_input"] = 0.9
            self.assertFalse(settings_mod._allow_numeric_update("top_p", "_top_p_input"))

        self.assertEqual(fake_st.session_state["_top_p_input"], 1.0)


if __name__ == "__main__":
    unittest.main()
