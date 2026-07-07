"""Tests for Conversation — specifically the JSONL persistence hook.

Uses a fake TurnRunner so no API calls happen; the Turn payloads are
fabricated the same way tests/test_turn.py does it.
"""

import json
from datetime import UTC, datetime

from anthropic.types import Message

from gene.agent.conversation import Conversation
from gene.agent.turn import Step, Turn
from gene.agent.turn_runner import TurnRunner


def _make_turn(user_input: str = "hi") -> Turn:
    msg = Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-x",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
    )
    now = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    step = Step(
        request={"model": "claude-x", "messages": []},
        response=msg,
        tool_calls=[],
        input_tokens=5,
        output_tokens=3,
        api_seconds=0.1,
        seconds=0.1,
        cache_hit=False,
        started_at=now,
        completed_at=now,
    )
    return Turn(
        id="abc123",
        user_input=user_input,
        steps=[step],
        new_messages=[
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ],
        final_message=msg,
        terminal_reason="end_turn",
        error=None,
        started_at=now,
        completed_at=now,
    )


class FakeRunner(TurnRunner):
    """Stand-in for TurnRunner: returns pre-built Turns, records the calls.

    Subclasses `TurnRunner` (rather than duck-typing) so `Conversation`'s
    static type check accepts it. We skip `TurnRunner.__init__` entirely
    to avoid pulling in an `llm` dependency; that's fine because `run`
    is overridden and none of the inherited attributes are touched.
    """

    def __init__(self, turns: list[Turn]):
        self._turns = list(turns)
        self.calls: list[tuple[list[dict], str, str | None]] = []

    def run(self, messages, user_input, system=None, tools=None, max_steps=10):
        self.calls.append((list(messages), user_input, system))
        return self._turns.pop(0)


def test_ask_without_log_path_writes_nothing(tmp_path):
    runner = FakeRunner([_make_turn()])
    conv = Conversation(runner, log_path=None)
    conv.ask("hi")
    assert list(tmp_path.iterdir()) == []


def test_ask_with_log_path_appends_one_line_per_turn(tmp_path):
    turns = [_make_turn("q1"), _make_turn("q2")]
    log = tmp_path / "session.jsonl"
    runner = FakeRunner(turns)
    conv = Conversation(runner, log_path=log)

    conv.ask("q1")
    conv.ask("q2")

    lines = log.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["user_input"] == "q1"
    assert json.loads(lines[1])["user_input"] == "q2"


def test_logged_turn_round_trips(tmp_path):
    original = _make_turn("what is 1+2?")
    log = tmp_path / "session.jsonl"
    conv = Conversation(FakeRunner([original]), log_path=log)
    conv.ask("what is 1+2?")

    restored = Turn.from_dict(json.loads(log.read_text().strip()))
    assert restored == original


def test_log_path_parent_dir_is_created(tmp_path):
    log = tmp_path / "nested" / "deeper" / "session.jsonl"
    conv = Conversation(FakeRunner([_make_turn()]), log_path=log)
    conv.ask("hi")
    assert log.exists()


def test_log_path_accepts_str(tmp_path):
    log = tmp_path / "session.jsonl"
    conv = Conversation(FakeRunner([_make_turn()]), log_path=str(log))
    conv.ask("hi")
    assert log.exists()


def test_ask_still_updates_history_and_turns_when_logging(tmp_path):
    log = tmp_path / "session.jsonl"
    conv = Conversation(FakeRunner([_make_turn("q")]), log_path=log)
    turn = conv.ask("q")
    assert conv.turns == [turn]
    assert conv.history == turn.new_messages
