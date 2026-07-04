"""Tests for gene.log_view.

Uses the same Turn-fabrication helper as test_conversation.py — no API calls.
"""

import json
from datetime import UTC, datetime

import pytest
from anthropic.types import Message

from gene.log_view import load_turns, main, render
from gene.turn import Step, Turn


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
        id="abc12345",
        user_input=user_input,
        steps=[step],
        new_messages=[],
        final_message=msg,
        terminal_reason="end_turn",
        error=None,
        started_at=now,
        completed_at=now,
    )


def _write_log(path, turns):
    path.write_text("".join(json.dumps(t.to_dict()) + "\n" for t in turns))


def test_load_turns_reads_all_lines(tmp_path):
    log = tmp_path / "log.jsonl"
    _write_log(log, [_make_turn("q1"), _make_turn("q2")])
    loaded = load_turns(log)
    assert [t.user_input for t in loaded] == ["q1", "q2"]


def test_load_turns_skips_blank_lines(tmp_path):
    log = tmp_path / "log.jsonl"
    good = json.dumps(_make_turn("q").to_dict())
    log.write_text(f"{good}\n\n{good}\n")
    assert len(load_turns(log)) == 2


def test_load_turns_skips_malformed_line(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    good = json.dumps(_make_turn("q").to_dict())
    log.write_text(f"{good}\nnot-json\n{good}\n")
    loaded = load_turns(log)
    assert len(loaded) == 2
    assert "skipping line 2" in capsys.readouterr().err


def test_render_default_shows_header_in_out(tmp_path):
    indexed = list(enumerate([_make_turn("q1"), _make_turn("q2")]))
    out = render(indexed, trace=False)
    assert "Turn number: 0" in out
    assert "Turn number: 1" in out
    assert "In:  q1" in out
    assert "In:  q2" in out
    assert "Out: hello" in out


def test_render_trace_has_header(tmp_path):
    indexed = list(enumerate([_make_turn("q1"), _make_turn("q2")]))
    out = render(indexed, trace=True)
    assert "Turn number: 0" in out
    assert "Turn number: 1" in out
    assert "'q1'" in out
    assert "'q2'" in out


def test_render_default_shows_terminal_reason_when_no_reply(tmp_path):
    turn = _make_turn("q")._replace(final_message=None, terminal_reason="max_steps")
    out = render([(0, turn)], trace=False)
    assert "(no reply — reason=max_steps)" in out


def test_render_default_truncates_long_content(tmp_path):
    long = "x" * 500
    out = render([(0, _make_turn(long))], trace=False)
    assert "..." in out
    assert "x" * 500 not in out


def test_main_default_prints_headline(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    _write_log(log, [_make_turn("q1"), _make_turn("q2")])
    main([str(log)])
    out = capsys.readouterr().out
    assert "Turn number: 0" in out
    assert "Turn number: 1" in out
    assert "In:  q1" in out
    assert "In:  q2" in out


def test_main_tail_preserves_original_index(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    _write_log(log, [_make_turn("q1"), _make_turn("q2"), _make_turn("q3")])
    main([str(log), "--tail", "2"])
    out = capsys.readouterr().out
    assert "Turn number: 1" in out
    assert "Turn number: 2" in out
    assert "Turn number: 0" not in out


def test_main_turn_preserves_original_index(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    _write_log(log, [_make_turn("q1"), _make_turn("q2"), _make_turn("q3")])
    main([str(log), "--turn", "-1"])
    out = capsys.readouterr().out
    assert "Turn number: 2" in out
    assert "In:  q3" in out


def test_main_turn_selects_single(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    _write_log(log, [_make_turn("q1"), _make_turn("q2"), _make_turn("q3")])
    main([str(log), "--turn", "1", "--trace"])
    out = capsys.readouterr().out
    assert "'q2'" in out
    assert "'q1'" not in out
    assert "'q3'" not in out


def test_main_turn_negative_index(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    _write_log(log, [_make_turn("q1"), _make_turn("q2"), _make_turn("q3")])
    main([str(log), "--turn", "-1", "--trace"])
    out = capsys.readouterr().out
    assert "'q3'" in out


def test_main_turn_out_of_range_exits(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    _write_log(log, [_make_turn("q1")])
    with pytest.raises(SystemExit) as exc:
        main([str(log), "--turn", "5"])
    assert exc.value.code == 1
    assert "out of range" in capsys.readouterr().err


def test_main_tail_limits(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    _write_log(log, [_make_turn("q1"), _make_turn("q2"), _make_turn("q3")])
    main([str(log), "--tail", "2"])
    out = capsys.readouterr().out
    assert "q1" not in out
    assert "In:  q2" in out
    assert "In:  q3" in out


def test_main_empty_file_prints_nothing(tmp_path, capsys):
    log = tmp_path / "empty.jsonl"
    log.write_text("")
    main([str(log)])
    assert capsys.readouterr().out == ""


def test_main_turn_and_tail_mutually_exclusive(tmp_path):
    log = tmp_path / "log.jsonl"
    _write_log(log, [_make_turn()])
    with pytest.raises(SystemExit):
        main([str(log), "--turn", "0", "--tail", "1"])
