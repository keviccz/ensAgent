from __future__ import annotations

import unittest
from concurrent.futures import Future
from unittest.mock import patch

from streamlit_app.components import chat as chat_mod


class _DummyAPI:
    def __init__(self, *args, **kwargs):
        self.calls = 0
        self.last_messages = []

    def initialize(self) -> bool:
        return True

    def chat(self, **kwargs):
        self.calls += 1
        self.last_messages = kwargs.get("messages", [])
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
            chat_mod, "_stream_text", side_effect=lambda text, _placeholder, stream=True: text
        ):
            out = chat_mod._generate_response_streaming("Show config and proceed", placeholder=object())

        self.assertIn("Tool execution finished.", out)
        self.assertEqual(dummy_api.calls, 2)
        mock_exec.assert_called_once()
        self.assertEqual(mock_exec.call_args[0][0], "show_config")

    def test_system_prompt_includes_runtime_parameters(self) -> None:
        state = {
            "api_model": "gpt-4o",
            "api_provider": "azure",
            "sample_id": "DLPFC_151507",
            "data_path": "Tool-runner/151507",
            "n_clusters": 5,
            "temperature": 0.9,
            "top_p": 0.8,
            "max_tokens": 1024,
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        dummy_api = _DummyAPI()

        with patch.object(chat_mod, "EnsAgentAPI", return_value=dummy_api), patch.object(
            chat_mod, "get_state", side_effect=_get_state
        ), patch.object(chat_mod, "load_config", return_value=object()), patch.object(
            chat_mod, "execute_tool", return_value={"ok": True}
        ), patch.object(
            chat_mod, "_stream_text", side_effect=lambda text, _placeholder, stream=True: text
        ):
            chat_mod._generate_response_streaming("show config", placeholder=None, stream=False)

        system_prompt = dummy_api.last_messages[0]["content"]
        self.assertIn("- n_clusters: 5", system_prompt)
        self.assertIn("- Temperature: 0.9", system_prompt)
        self.assertIn("- Top-p      : 0.8", system_prompt)

    def test_process_pending_response_submits_background_job(self) -> None:
        state = {
            "pending_user_input": "hello",
            "pending_request_conversation_id": "conv_1",
            "pending_response_job_id": None,
            "pending_response_inflight": False,
            "current_conversation_id": "conv_1",
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        def _set_state(key, value):
            state[key] = value

        with patch.object(chat_mod, "get_state", side_effect=_get_state), patch.object(
            chat_mod, "set_state", side_effect=_set_state
        ), patch.object(chat_mod, "_submit_response_job", return_value="job_123") as mock_submit:
            chat_mod.process_pending_response()

        mock_submit.assert_called_once()
        self.assertEqual(state["pending_response_job_id"], "job_123")
        self.assertTrue(state["pending_response_inflight"])

    def test_process_pending_response_completes_finished_job(self) -> None:
        done = Future()
        done.set_result("assistant done")
        state = {
            "pending_user_input": "hello",
            "pending_request_conversation_id": "conv_1",
            "pending_response_job_id": "job_123",
            "pending_response_inflight": True,
            "current_conversation_id": "conv_1",
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        def _set_state(key, value):
            state[key] = value

        with patch.object(chat_mod, "get_state", side_effect=_get_state), patch.object(
            chat_mod, "set_state", side_effect=_set_state
        ), patch.object(chat_mod, "_get_response_job", return_value=done), patch.object(
            chat_mod, "_clear_response_job"
        ) as mock_clear, patch.object(chat_mod, "add_message") as mock_add:
            chat_mod.process_pending_response()

        mock_clear.assert_called_once_with("job_123")
        mock_add.assert_called_once_with("assistant", "assistant done")
        self.assertIsNone(state["pending_user_input"])
        self.assertIsNone(state["pending_request_conversation_id"])
        self.assertIsNone(state["pending_response_job_id"])
        self.assertFalse(state["pending_response_inflight"])


if __name__ == "__main__":
    unittest.main()
