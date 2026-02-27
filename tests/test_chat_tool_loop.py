from __future__ import annotations

import unittest
from unittest.mock import patch

from streamlit_app.components import chat as chat_mod


class _DummyAPI:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def initialize(self) -> bool:
        return True

    def chat(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "show_config",
                        "arguments": {},
                    }
                ],
            }
        return {"content": "Tool execution finished.", "tool_calls": []}


class ChatToolLoopTests(unittest.TestCase):
    def test_generate_response_executes_tool_calls(self) -> None:
        state = {
            "api_model": "gpt-4o",
            "api_provider": "azure",
            "sample_id": "DLPFC_151507",
            "data_path": "Tool-runner/151507",
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 1024,
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        dummy_api = _DummyAPI()

        with patch.object(chat_mod, "EnsAgentAPI", return_value=dummy_api), patch.object(
            chat_mod, "get_state", side_effect=_get_state
        ), patch.object(chat_mod, "load_config", return_value=object()), patch.object(
            chat_mod, "execute_tool", return_value={"ok": True}
        ) as mock_exec, patch.object(
            chat_mod, "_stream_text", side_effect=lambda text, _placeholder: text
        ):
            out = chat_mod._generate_response_streaming("Show config and proceed", placeholder=object())

        self.assertIn("Tool execution finished.", out)
        self.assertEqual(dummy_api.calls, 2)
        mock_exec.assert_called_once()
        self.assertEqual(mock_exec.call_args[0][0], "show_config")


if __name__ == "__main__":
    unittest.main()
