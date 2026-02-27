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


if __name__ == "__main__":
    unittest.main()
