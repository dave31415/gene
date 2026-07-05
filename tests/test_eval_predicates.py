"""Tests for the small predicate helpers used by eval-case checks.

Fabricates minimal Turn/Step/ToolCall values — no API or DB — so the
predicates can be exercised in isolation. Text-only predicates live in
`eval_predicates`; Turn-based ones in `eval_predicates_turn`.
"""

from datetime import UTC, datetime

from anthropic.types import Message

from gene.agent.eval_predicates import contains_all, contains_any
from gene.agent.eval_predicates_turn import max_steps, sql_matches, used_tool
from gene.agent.tool import ToolCall
from gene.agent.turn import Step, Turn


def _msg(text: str = "hi") -> Message:
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-x",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
    )


def _step(tool_calls: list[ToolCall] | None = None) -> Step:
    now = datetime(2026, 7, 4, tzinfo=UTC)
    return Step(
        request={"messages": []},
        response=_msg(),
        tool_calls=tool_calls or [],
        input_tokens=1,
        output_tokens=1,
        api_seconds=0.0,
        seconds=0.0,
        cache_hit=False,
        started_at=now,
        completed_at=now,
    )


def _turn(steps: list[Step] | None = None) -> Turn:
    now = datetime(2026, 7, 4, tzinfo=UTC)
    return Turn(
        id="abc",
        user_input="q",
        steps=steps or [_step()],
        new_messages=[],
        final_message=_msg(),
        terminal_reason="end_turn",
        error=None,
        started_at=now,
        completed_at=now,
    )


def _tc(name: str = "run_query", sql: str = "SELECT 1") -> ToolCall:
    return ToolCall(
        tool_use_id="tu_1",
        name=name,
        input={"sql": sql},
        output="{}",
        is_error=False,
        seconds=0.0,
    )


# ---------- text predicates ----------


def test_contains_all_true_when_every_needle_present():
    assert contains_all("John Dennis and Mary Catherine", ["John Dennis", "Mary Catherine"])


def test_contains_all_false_when_one_missing():
    assert not contains_all("John Dennis only", ["John Dennis", "Mary Catherine"])


def test_contains_any_true_when_one_matches():
    assert contains_any("Just John here", ["John", "Mary"])


def test_contains_any_false_when_none_match():
    assert not contains_any("nothing to see", ["John", "Mary"])


# ---------- turn predicates ----------


def test_max_steps_true_at_or_below_cap():
    assert max_steps(_turn(steps=[_step(), _step()]), 2)
    assert max_steps(_turn(steps=[_step()]), 2)


def test_max_steps_false_above_cap():
    assert not max_steps(_turn(steps=[_step(), _step(), _step()]), 2)


def test_used_tool_matches_name():
    turn = _turn(steps=[_step(tool_calls=[_tc(name="run_query")])])
    assert used_tool(turn, "run_query")
    assert not used_tool(turn, "calculator")


def test_used_tool_false_when_no_calls():
    assert not used_tool(_turn(), "run_query")


def test_sql_matches_finds_pattern_case_insensitive():
    turn = _turn(steps=[_step(tool_calls=[_tc(sql="SELECT * FROM t WHERE sex = 'M'")])])
    assert sql_matches(turn, r"\bsex\s*=")
    assert sql_matches(turn, r"\bwhere\b")


def test_sql_matches_false_when_pattern_absent():
    turn = _turn(steps=[_step(tool_calls=[_tc(sql="SELECT COUNT(*) FROM t")])])
    assert not sql_matches(turn, r"\bwhere\b")


def test_sql_matches_scans_all_tool_calls():
    turn = _turn(
        steps=[
            _step(tool_calls=[_tc(sql="SELECT 1"), _tc(sql="SELECT * WHERE surname='X'")]),
        ]
    )
    assert sql_matches(turn, r"\bsurname\s*=")
