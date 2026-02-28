from __future__ import annotations

from datetime import datetime, timedelta
import unittest
from unittest.mock import patch

from streamlit_app.utils import state as state_mod


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _FakeStreamlit:
    def __init__(self, session_state):
        self.session_state = session_state


class StateHistoryOrderTests(unittest.TestCase):
    def _build_fake_state(self) -> _SessionState:
        return _SessionState(
            {
                "messages": [],
                "conversations": [],
                "current_conversation_id": None,
            }
        )

    def test_load_conversation_does_not_touch_previous_updated_at(self) -> None:
        session = self._build_fake_state()
        now = datetime.now()
        conv_a = state_mod.Conversation(
            id="a1",
            title="A",
            messages=[state_mod.ChatMessage(role="user", content="old")],
            created_at=now - timedelta(hours=3),
            updated_at=now - timedelta(hours=2),
        )
        conv_b = state_mod.Conversation(
            id="b2",
            title="B",
            messages=[state_mod.ChatMessage(role="user", content="new")],
            created_at=now - timedelta(hours=1),
            updated_at=now - timedelta(minutes=30),
        )
        previous_updated = conv_a.updated_at
        session.conversations = [conv_a, conv_b]
        session.current_conversation_id = conv_a.id
        session.messages = conv_a.messages.copy()

        with patch.object(state_mod, "st", _FakeStreamlit(session)), patch.object(
            state_mod, "persist_conversation"
        ):
            loaded = state_mod.load_conversation(conv_b.id)

        self.assertTrue(loaded)
        self.assertEqual(conv_a.updated_at, previous_updated)
        self.assertEqual(session.current_conversation_id, conv_b.id)

    def test_new_message_promotes_conversation_to_top(self) -> None:
        session = self._build_fake_state()
        now = datetime.now()
        older_conv = state_mod.Conversation(
            id="old1",
            title="Old",
            messages=[],
            created_at=now - timedelta(hours=4),
            updated_at=now - timedelta(hours=4),
        )
        newer_conv = state_mod.Conversation(
            id="new2",
            title="Newer",
            messages=[state_mod.ChatMessage(role="user", content="latest")],
            created_at=now - timedelta(hours=1),
            updated_at=now - timedelta(hours=1),
        )
        session.conversations = [older_conv, newer_conv]
        session.current_conversation_id = older_conv.id
        session.messages = []

        with patch.object(state_mod, "st", _FakeStreamlit(session)), patch.object(
            state_mod, "persist_conversation"
        ):
            state_mod.add_message("user", "promote this conversation")
            ordered = state_mod.get_conversations()

        self.assertEqual(ordered[0].id, older_conv.id)

    def test_start_new_conversation_is_first(self) -> None:
        session = self._build_fake_state()
        now = datetime.now()
        session.conversations = [
            state_mod.Conversation(
                id="x1",
                title="X",
                messages=[state_mod.ChatMessage(role="user", content="x")],
                created_at=now - timedelta(hours=2),
                updated_at=now - timedelta(hours=2),
            ),
            state_mod.Conversation(
                id="y2",
                title="Y",
                messages=[state_mod.ChatMessage(role="user", content="y")],
                created_at=now - timedelta(hours=1),
                updated_at=now - timedelta(hours=1),
            ),
        ]
        session.current_conversation_id = "y2"
        session.messages = []

        with patch.object(state_mod, "st", _FakeStreamlit(session)), patch.object(
            state_mod, "persist_conversation"
        ):
            new_id = state_mod.start_new_conversation()
            ordered = state_mod.get_conversations()

        self.assertEqual(session.current_conversation_id, new_id)
        self.assertEqual(ordered[0].id, new_id)

    def test_start_new_conversation_reuses_active_empty_conversation(self) -> None:
        session = self._build_fake_state()
        now = datetime.now()
        empty = state_mod.Conversation(
            id="empty1",
            title="New conversation",
            messages=[],
            created_at=now - timedelta(minutes=5),
            updated_at=now - timedelta(minutes=5),
        )
        session.conversations = [empty]
        session.current_conversation_id = empty.id
        session.messages = []

        with patch.object(state_mod, "st", _FakeStreamlit(session)), patch.object(
            state_mod, "persist_conversation"
        ), patch.object(state_mod, "delete_conversation_from_disk") as mock_del:
            reused_id = state_mod.start_new_conversation()

        self.assertEqual(reused_id, empty.id)
        self.assertEqual(len(session.conversations), 1)
        mock_del.assert_not_called()

    def test_start_new_conversation_prunes_old_empty_conversations(self) -> None:
        session = self._build_fake_state()
        now = datetime.now()
        active = state_mod.Conversation(
            id="active1",
            title="Active",
            messages=[state_mod.ChatMessage(role="user", content="hello")],
            created_at=now - timedelta(minutes=10),
            updated_at=now - timedelta(minutes=10),
        )
        old_empty = state_mod.Conversation(
            id="old_empty",
            title="New conversation",
            messages=[],
            created_at=now - timedelta(minutes=15),
            updated_at=now - timedelta(minutes=15),
        )
        session.conversations = [old_empty, active]
        session.current_conversation_id = active.id
        session.messages = active.messages.copy()

        with patch.object(state_mod, "st", _FakeStreamlit(session)), patch.object(
            state_mod, "persist_conversation"
        ), patch.object(state_mod, "delete_conversation_from_disk") as mock_del:
            new_id = state_mod.start_new_conversation()

        self.assertEqual(session.current_conversation_id, new_id)
        self.assertNotIn("old_empty", [c.id for c in session.conversations])
        self.assertEqual(len([c for c in session.conversations if len(c.messages) == 0]), 1)
        mock_del.assert_called_once_with("old_empty")

    def test_init_session_state_includes_pending_chat_keys(self) -> None:
        session = _SessionState({})
        with patch.object(state_mod, "st", _FakeStreamlit(session)), patch.object(
            state_mod, "load_conversations_from_disk", return_value=[]
        ):
            state_mod.init_session_state()

        self.assertIn("pending_user_input", session)
        self.assertIn("pending_request_conversation_id", session)
        self.assertIn("pending_response_inflight", session)
        self.assertIn("pending_response_job_id", session)
        self.assertIsNone(session["pending_user_input"])
        self.assertFalse(session["pending_response_inflight"])
        self.assertIsNone(session["pending_response_job_id"])


if __name__ == "__main__":
    unittest.main()
