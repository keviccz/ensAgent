from __future__ import annotations

import unittest
from concurrent.futures import Future
from pathlib import Path
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


class _SimpleCfg:
    data_path = ""
    sample_id = ""
    csv_path = ""
    n_clusters = 7
    temperature = 0.7
    top_p = 1.0

    def resolved_scoring_input_dir(self) -> Path:
        return Path("scoring") / "input"


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

    def test_explicit_stage_b_request_executes_scoring_directly(self) -> None:
        state = {
            "api_model": "gpt-4o",
            "api_provider": "azure",
            "sample_id": "DLPFC_151507",
            "data_path": "Tool-runner/151507",
            "csv_path": "tests",
            "n_clusters": 7,
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 1024,
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        cfg = _SimpleCfg()
        cfg.sample_id = state["sample_id"]
        cfg.data_path = state["data_path"]
        cfg.csv_path = state["csv_path"]

        with patch.object(chat_mod, "get_state", side_effect=_get_state), patch.object(
            chat_mod, "load_config", return_value=cfg
        ), patch.object(chat_mod, "execute_tool", return_value={"ok": True, "exit_code": 0}) as mock_exec, patch.object(
            chat_mod, "EnsAgentAPI"
        ) as mock_api:
            out = chat_mod._generate_response_streaming("Please run Stage B now", placeholder=None, stream=False)

        self.assertIn("run_scoring", out)
        mock_exec.assert_called_once()
        self.assertEqual(mock_exec.call_args[0][0], "run_scoring")
        self.assertEqual(mock_exec.call_args[0][1]["input_dir"], "tests")
        mock_api.assert_not_called()

    def test_explicit_stage_a_request_reports_missing_required_config(self) -> None:
        state = {
            "api_model": "gpt-4o",
            "api_provider": "azure",
            "sample_id": "",
            "data_path": "",
            "csv_path": "tests",
            "n_clusters": 7,
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 1024,
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        cfg = _SimpleCfg()
        with patch.object(chat_mod, "get_state", side_effect=_get_state), patch.object(
            chat_mod, "load_config", return_value=cfg
        ), patch.object(chat_mod, "execute_tool") as mock_exec, patch.object(
            chat_mod, "EnsAgentAPI"
        ) as mock_api:
            out = chat_mod._generate_response_streaming("Run Stage A now", placeholder=None, stream=False)

        self.assertIn("data_path", out)
        self.assertIn("sample_id", out)
        mock_exec.assert_not_called()
        mock_api.assert_not_called()

    def test_stage_b_question_does_not_force_direct_execution(self) -> None:
        state = {
            "api_model": "gpt-4o",
            "api_provider": "azure",
            "sample_id": "DLPFC_151507",
            "data_path": "Tool-runner/151507",
            "csv_path": "tests",
            "n_clusters": 7,
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 1024,
        }

        class _InfoAPI:
            def __init__(self, *args, **kwargs):
                self.calls = 0

            def initialize(self) -> bool:
                return True

            def chat(self, **kwargs):
                self.calls += 1
                return {"content": "Stage B handles scoring.", "tool_calls": []}

        def _get_state(key, default=None):
            return state.get(key, default)

        cfg = _SimpleCfg()
        info_api = _InfoAPI()
        with patch.object(chat_mod, "EnsAgentAPI", return_value=info_api), patch.object(
            chat_mod, "get_state", side_effect=_get_state
        ), patch.object(chat_mod, "load_config", return_value=cfg), patch.object(
            chat_mod, "execute_tool"
        ) as mock_exec:
            out = chat_mod._generate_response_streaming("What is Stage B?", placeholder=None, stream=False)

        self.assertIn("Stage B handles scoring.", out)
        self.assertEqual(info_api.calls, 1)
        mock_exec.assert_not_called()

    def test_process_pending_response_submits_background_job(self) -> None:
        state = {
            "pending_user_input": "hello",
            "pending_request_conversation_id": "conv_1",
            "pending_request_id": "req_1",
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
        self.assertEqual(mock_submit.call_args[0][2], "req_1")
        self.assertEqual(state["pending_response_job_id"], "job_123")
        self.assertTrue(state["pending_response_inflight"])

    def test_process_pending_response_completes_finished_job(self) -> None:
        done = Future()
        done.set_result("assistant done")
        state = {
            "pending_user_input": "hello",
            "pending_request_conversation_id": "conv_1",
            "pending_request_id": "req_1",
            "pending_response_job_id": "job_123",
            "pending_response_inflight": True,
            "current_conversation_id": "conv_1",
            "_chat_completion_tick": 0,
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        def _set_state(key, value):
            state[key] = value

        with patch.object(chat_mod, "get_state", side_effect=_get_state), patch.object(
            chat_mod, "set_state", side_effect=_set_state
        ), patch.object(
            chat_mod,
            "_get_response_job",
            return_value=chat_mod.ResponseJob(future=done, cancel_event=None, request_id="req_1"),
        ), patch.object(
            chat_mod, "_clear_response_job"
        ) as mock_clear, patch.object(chat_mod, "add_message") as mock_add:
            chat_mod.process_pending_response()

        mock_clear.assert_called_once_with("job_123")
        mock_add.assert_called_once_with("assistant", "assistant done")
        self.assertIsNone(state["pending_user_input"])
        self.assertIsNone(state["pending_request_conversation_id"])
        self.assertIsNone(state["pending_request_id"])
        self.assertIsNone(state["pending_response_job_id"])
        self.assertFalse(state["pending_response_inflight"])
        self.assertEqual(state["_chat_completion_tick"], 1)

    def test_process_pending_response_ignores_stale_request_result(self) -> None:
        done = Future()
        done.set_result("stale result")
        state = {
            "pending_user_input": "hello",
            "pending_request_conversation_id": "conv_1",
            "pending_request_id": "req_new",
            "pending_response_job_id": "job_123",
            "pending_response_inflight": True,
            "current_conversation_id": "conv_1",
            "_chat_completion_tick": 0,
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        def _set_state(key, value):
            state[key] = value

        with patch.object(chat_mod, "get_state", side_effect=_get_state), patch.object(
            chat_mod, "set_state", side_effect=_set_state
        ), patch.object(
            chat_mod, "_get_response_job", return_value=chat_mod.ResponseJob(future=done, cancel_event=None, request_id="req_old")
        ), patch.object(chat_mod, "_clear_response_job") as mock_clear, patch.object(
            chat_mod, "add_message"
        ) as mock_add:
            chat_mod.process_pending_response()

        mock_clear.assert_called_once_with("job_123")
        mock_add.assert_not_called()
        self.assertFalse(state["pending_response_inflight"])
        self.assertEqual(state["_chat_completion_tick"], 0)

    def test_should_show_quick_actions_when_not_hidden(self) -> None:
        state = {
            "current_conversation_id": "conv_1",
            "_quick_actions_hidden_conversations": [],
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        with patch.object(chat_mod, "get_state", side_effect=_get_state):
            self.assertTrue(chat_mod._should_show_quick_actions([], False))

    def test_should_hide_quick_actions_when_conversation_marked(self) -> None:
        state = {
            "current_conversation_id": "conv_1",
            "_quick_actions_hidden_conversations": ["conv_1"],
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        with patch.object(chat_mod, "get_state", side_effect=_get_state):
            self.assertFalse(chat_mod._should_show_quick_actions([], False))

    def test_hide_quick_actions_marks_current_conversation(self) -> None:
        state = {
            "current_conversation_id": "conv_1",
            "_quick_actions_hidden_conversations": [],
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        def _set_state(key, value):
            state[key] = value

        with patch.object(chat_mod, "get_state", side_effect=_get_state), patch.object(
            chat_mod, "set_state", side_effect=_set_state
        ):
            chat_mod._hide_quick_actions_for_current_conversation()
            chat_mod._hide_quick_actions_for_current_conversation()

        self.assertEqual(state["_quick_actions_hidden_conversations"], ["conv_1"])

    def test_chat_ui_pending_activity_includes_pending_prompt(self) -> None:
        state = {
            "pending_user_input": None,
            "pending_response_inflight": False,
            "pending_prompt": "Run end-to-end analysis",
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        with patch.object(chat_mod, "get_state", side_effect=_get_state):
            self.assertTrue(chat_mod._chat_ui_has_pending_activity())

    def test_queue_quick_action_creates_conversation_and_hides_actions(self) -> None:
        state = {"current_conversation_id": None}

        def _get_state(key, default=None):
            return state.get(key, default)

        def _set_state(key, value):
            state[key] = value

        with patch.object(chat_mod, "get_state", side_effect=_get_state), patch.object(
            chat_mod, "set_state", side_effect=_set_state
        ), patch.object(chat_mod, "start_new_conversation", side_effect=lambda: state.update(current_conversation_id="conv_new")), patch.object(
            chat_mod, "_hide_quick_actions_for_current_conversation"
        ) as mock_hide:
            chat_mod._queue_quick_action_prompt("Check environment status")

        self.assertEqual(state["pending_prompt"], "Check environment status")
        self.assertEqual(state["current_conversation_id"], "conv_new")
        mock_hide.assert_called_once()

    def test_consume_pending_action_routes_to_chat_prompt(self) -> None:
        state = {
            "pending_action": "check_envs",
            "active_page": "spatial",
            "pending_prompt": None,
        }

        def _get_state(key, default=None):
            return state.get(key, default)

        def _set_state(key, value):
            state[key] = value

        with patch.object(chat_mod, "get_state", side_effect=_get_state), patch.object(
            chat_mod, "set_state", side_effect=_set_state
        ):
            chat_mod.consume_pending_action_prompt()

        self.assertEqual(state["pending_action"], None)
        self.assertEqual(state["active_page"], "chat")
        self.assertEqual(state["pending_prompt"], "Check environment status")

    def test_execute_tool_compat_falls_back_for_legacy_signature(self) -> None:
        chat_mod._EXECUTE_TOOL_REBIND_ATTEMPTED = False
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        progress_events: list[dict] = []

        def _legacy_execute_tool(*args, **kwargs):
            if kwargs:
                raise TypeError("execute_tool() got an unexpected keyword argument 'progress_callback'")
            calls.append((args, kwargs))
            return {"ok": True, "legacy": True}

        with patch.object(chat_mod, "execute_tool", side_effect=_legacy_execute_tool):
            out = chat_mod._execute_tool_compat(
                "run_scoring",
                {"input_dir": "tests"},
                object(),
                progress_callback=lambda event: progress_events.append(event),
                cancel_check=lambda: False,
            )

        self.assertEqual(out, {"ok": True, "legacy": True})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], "run_scoring")
        self.assertEqual(progress_events, [])


if __name__ == "__main__":
    unittest.main()
