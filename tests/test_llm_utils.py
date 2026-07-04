"""Unit tests for gene.llm_utils.

No mocks — pure functions test directly, filesystem-touching code uses
pytest's built-in `tmp_path`, and `Cache` is exercised end-to-end with
plain Python functions as the "external" call.
"""

import json
import logging
from pathlib import Path

import pytest

from gene.llm_utils import (
    Cache,
    LlmConfigError,
    build_request,
    cache_key,
    resolve_api_key,
)

# ---------- _resolve_api_key ----------


def test_resolve_api_key_returns_stripped_content(tmp_path):
    # uses the Pytest built in fixture tmp_path
    (tmp_path / "anthropic").write_text("sk-test-123\n")
    assert resolve_api_key(tmp_path) == "sk-test-123"


def test_resolve_api_key_strips_surrounding_whitespace(tmp_path):
    (tmp_path / "anthropic").write_text("   sk-test-abc   \n")
    assert resolve_api_key(tmp_path) == "sk-test-abc"


def test_resolve_api_key_raises_when_file_missing(tmp_path):
    with pytest.raises(LlmConfigError) as exc:
        resolve_api_key(tmp_path)
    assert str(tmp_path / "anthropic") in str(exc.value)


def test_resolve_api_key_raises_when_file_empty(tmp_path):
    (tmp_path / "anthropic").write_text("")
    with pytest.raises(LlmConfigError, match="empty"):
        resolve_api_key(tmp_path)


def test_resolve_api_key_raises_when_file_whitespace_only(tmp_path):
    (tmp_path / "anthropic").write_text("   \n\t\n")
    with pytest.raises(LlmConfigError, match="empty"):
        resolve_api_key(tmp_path)


# ---------- _cache_key ----------


def test_cache_key_is_deterministic():
    req = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert cache_key(req) == cache_key(req)


def test_cache_key_ignores_top_level_dict_order():
    a = {"model": "m", "max_tokens": 10}
    b = {"max_tokens": 10, "model": "m"}
    assert cache_key(a) == cache_key(b)


def test_cache_key_ignores_nested_dict_order():
    a = {"messages": [{"role": "user", "content": "hi"}]}
    b = {"messages": [{"content": "hi", "role": "user"}]}
    assert cache_key(a) == cache_key(b)


def test_cache_key_changes_when_model_changes():
    base = {"model": "m", "messages": []}
    assert cache_key(base) != cache_key({**base, "model": "n"})


def test_cache_key_changes_when_messages_change():
    base = {"model": "m", "messages": [{"role": "user", "content": "a"}]}
    other = {"model": "m", "messages": [{"role": "user", "content": "b"}]}
    assert cache_key(base) != cache_key(other)


def test_cache_key_handles_non_json_types():
    # `default=str` lets Path (and other non-serializable types) through.
    cache_key({"path": Path("/tmp/x")})


def test_cache_key_is_sha256_hex():
    key = cache_key({"x": 1})
    assert len(key) == 64
    int(key, 16)  # parses as hex


# ---------- _build_request ----------

CFG = {"model": "claude-x", "max_tokens": 100, "temperature": None}


def test_build_request_minimal_shape():
    req = build_request(CFG, messages=[{"role": "user", "content": "hi"}])
    assert set(req) == {"model", "max_tokens", "messages"}
    assert req["model"] == "claude-x"
    assert req["max_tokens"] == 100
    assert req["messages"] == [{"role": "user", "content": "hi"}]


def test_build_request_omits_temperature_when_none():
    req = build_request(CFG, messages=[])
    assert "temperature" not in req


def test_build_request_includes_temperature_when_set():
    cfg = {**CFG, "temperature": 0.5}
    req = build_request(cfg, messages=[])
    assert req["temperature"] == 0.5


def test_build_request_omits_system_when_none():
    req = build_request(CFG, messages=[])
    assert "system" not in req


def test_build_request_includes_system_string():
    req = build_request(CFG, messages=[], system="you are helpful")
    assert req["system"] == "you are helpful"


def test_build_request_includes_system_list_form():
    blocks = [{"type": "text", "text": "you are helpful"}]
    req = build_request(CFG, messages=[], system=blocks)
    assert req["system"] == blocks


def test_build_request_omits_tools_when_none():
    req = build_request(CFG, messages=[])
    assert "tools" not in req


def test_build_request_includes_tools_when_set():
    tools = [{"name": "search", "input_schema": {"type": "object"}}]
    req = build_request(CFG, messages=[], tools=tools)
    assert req["tools"] == tools


def test_build_request_includes_tool_choice_when_set():
    choice = {"type": "any"}
    req = build_request(CFG, messages=[], tool_choice=choice)
    assert req["tool_choice"] == choice


def test_build_request_kitchen_sink():
    cfg = {**CFG, "temperature": 0.2}
    req = build_request(
        cfg,
        messages=[{"role": "user", "content": "hi"}],
        system="s",
        tools=[{"name": "t"}],
        tool_choice={"type": "auto"},
    )
    assert set(req) == {
        "model",
        "max_tokens",
        "messages",
        "temperature",
        "system",
        "tools",
        "tool_choice",
    }


# ---------- Cache ----------


def _json_cache(cache_dir, **kw):
    return Cache(cache_dir, serialize=json.dumps, deserialize=json.loads, **kw)


def test_cache_miss_then_hit_only_calls_func_once(tmp_path):
    calls = []

    def double(arg):
        calls.append(arg)
        return {"v": arg["n"] * 2}

    cache = _json_cache(tmp_path)
    v1, hit1 = cache.call(double, {"n": 5})
    v2, hit2 = cache.call(double, {"n": 5})
    assert v1 == v2 == {"v": 10}
    assert (hit1, hit2) == (False, True)
    assert calls == [{"n": 5}]


def test_cache_different_args_are_separate_entries(tmp_path):
    calls = []

    def double(arg):
        calls.append(arg)
        return {"v": arg["n"] * 2}

    cache = _json_cache(tmp_path)
    cache.call(double, {"n": 1})
    cache.call(double, {"n": 2})
    cache.call(double, {"n": 1})
    assert calls == [{"n": 1}, {"n": 2}]


def test_cache_key_order_invariance_e2e(tmp_path):
    """Cache hit must be insensitive to dict key order (guards _cache_key wiring)."""
    calls = []

    def f(arg):
        calls.append(arg)
        return "same"

    cache = _json_cache(tmp_path)
    cache.call(f, {"a": 1, "b": 2})
    cache.call(f, {"b": 2, "a": 1})
    assert calls == [{"a": 1, "b": 2}]


def test_cache_use_cache_false_bypasses_store(tmp_path):
    calls = []

    def f(arg):
        calls.append(arg)
        return arg

    cache = _json_cache(tmp_path, use_cache=False)
    cache.call(f, {"n": 1})
    cache.call(f, {"n": 1})
    assert calls == [{"n": 1}, {"n": 1}]
    assert cache._store is None


def test_cache_persists_across_instances(tmp_path):
    calls = []

    def f(arg):
        calls.append(arg)
        return {"v": arg["n"]}

    c1 = _json_cache(tmp_path)
    c1.call(f, {"n": 42})

    c2 = _json_cache(tmp_path)
    value, hit = c2.call(f, {"n": 42})
    assert value == {"v": 42}
    assert hit is True
    assert calls == [{"n": 42}]  # second Cache reused the on-disk entry


def test_cache_uses_serialize_on_write_and_deserialize_on_read(tmp_path):
    events = []

    def ser(v):
        events.append("ser")
        return json.dumps(v)

    def des(s):
        events.append("des")
        return json.loads(s)

    cache = Cache(tmp_path, serialize=ser, deserialize=des)
    cache.call(lambda arg: {"v": 1}, {"n": 1})  # miss  -> serialize
    cache.call(lambda arg: {"v": 1}, {"n": 1})  # hit   -> deserialize

    assert events == ["ser", "des"]


def test_cache_deserialize_returns_fresh_object(tmp_path):
    """A hit must round-trip through disk, not return the in-memory reference."""
    original = {"v": [1, 2, 3]}
    cache = _json_cache(tmp_path)
    cache.call(lambda _: original, {"n": 1})
    value, was_hit = cache.call(lambda _: original, {"n": 1})

    assert was_hit is True
    assert value == original
    assert value is not original  # went through serialize/deserialize
    assert value["v"] is not original["v"]


def test_cache_reports_miss_when_use_cache_false(tmp_path):
    cache = _json_cache(tmp_path, use_cache=False)
    _, hit = cache.call(lambda arg: {"v": 1}, {"n": 1})
    assert hit is False


def test_cache_propagates_exception_and_does_not_store(tmp_path):
    def boom(_):
        raise RuntimeError("nope")

    cache = _json_cache(tmp_path)
    with pytest.raises(RuntimeError, match="nope"):
        cache.call(boom, {"n": 1})

    # subsequent call still hits the failing func — nothing was cached
    with pytest.raises(RuntimeError):
        cache.call(boom, {"n": 1})


def test_cache_verbose_logs_miss_and_hit(tmp_path, caplog):
    cache = _json_cache(tmp_path, verbose=True)
    with caplog.at_level(logging.DEBUG, logger="gene.llm_utils"):
        cache.call(lambda _: {"v": 1}, {"n": 1})  # miss
        cache.call(lambda _: {"v": 1}, {"n": 1})  # hit

    msgs = [r.message for r in caplog.records]
    assert any("cache miss" in m for m in msgs)
    assert any("cache hit" in m for m in msgs)
