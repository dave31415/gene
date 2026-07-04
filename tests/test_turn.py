"""Round-trip serialization tests for Turn / Step.

Fabricates a minimal `anthropic.types.Message` via `Message.model_validate`
so we don't need a live API call to exercise the code paths that touch
Pydantic types.
"""

import json
from datetime import UTC, datetime

from anthropic.types import Message

from gene.agent.tool import ToolCall
from gene.agent.turn import Step, Turn, TurnError


def _make_message(*, stop_reason: str = "end_turn") -> Message:
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-x",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
    )


def _make_step() -> Step:
    return Step(
        request={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
        response=_make_message(),
        tool_calls=[
            ToolCall(
                tool_use_id="tu_1",
                name="calculator",
                input={"operation": "add", "a": 1, "b": 2},
                output="3",
                is_error=False,
                seconds=0.001,
            )
        ],
        input_tokens=5,
        output_tokens=3,
        api_seconds=0.42,
        seconds=0.43,
        cache_hit=False,
        started_at=datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC),
        completed_at=datetime(2026, 7, 3, 12, 0, 1, tzinfo=UTC),
    )


def _make_turn(*, with_error: bool = False) -> Turn:
    step = _make_step()
    return Turn(
        id="abc123",
        user_input="what is 1+2?",
        steps=[step],
        new_messages=[
            {"role": "user", "content": "what is 1+2?"},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ],
        final_message=step.response,
        terminal_reason="end_turn",
        error=TurnError(type="RuntimeError", message="boom", step_index=0) if with_error else None,
        started_at=datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC),
        completed_at=datetime(2026, 7, 3, 12, 0, 2, tzinfo=UTC),
    )


def test_turn_round_trip_equals_original():
    original = _make_turn()
    restored = Turn.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original


def test_turn_round_trip_with_error():
    original = _make_turn(with_error=True)
    restored = Turn.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.error is not None
    assert restored.error.type == "RuntimeError"


def test_turn_round_trip_with_no_final_message():
    turn = _make_turn()._replace(final_message=None, terminal_reason="max_steps")
    restored = Turn.from_dict(json.loads(json.dumps(turn.to_dict())))
    assert restored == turn
    assert restored.final_message is None


def test_step_round_trip_equals_original():
    original = _make_step()
    restored = Step.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
