"""One send-and-execute round trip with the LLM.

`execute_step` is the only place we talk to the model: send the current
messages, and if the response is a `tool_use`, run each requested tool
locally. Returns a `Step` — the observability record for that round.

`execute_tool` runs one tool_use block against a handler registry.
Kept in this file because you'd only ever call it from `execute_step`:
the tools it runs are named by the model's response.

Both functions are free — they take everything they need as arguments.
The `TurnRunner` holds the wiring (llm, schemas, handlers) and calls
in, but nothing here reads from `self`.
"""

import time
from datetime import UTC, datetime
from typing import Any

from gene.agent.llm import CachedAnthropic
from gene.agent.tool import Tool, ToolCall
from gene.agent.turn import Step


def execute_step(
    llm: CachedAnthropic,
    messages: list[dict[str, Any]],
    system: str | None,
    schemas: list[dict[str, Any]],
    handlers: dict[str, Tool],
) -> Step:
    """Send once, execute any tool_use blocks, return a Step record."""
    tools_arg = schemas if schemas else None
    started_at = datetime.now(UTC)
    t_step = time.perf_counter()

    t_api = time.perf_counter()
    msg, meta = llm.send(messages=messages, system=system, tools=tools_arg)
    api_seconds = time.perf_counter() - t_api

    tool_calls: list[ToolCall] = []
    if msg.stop_reason == "tool_use":
        tool_calls = [execute_tool(b, handlers) for b in msg.content if b.type == "tool_use"]

    seconds = time.perf_counter() - t_step
    return Step(
        request=meta["request"],
        response=msg,
        tool_calls=tool_calls,
        input_tokens=msg.usage.input_tokens,
        output_tokens=msg.usage.output_tokens,
        api_seconds=api_seconds,
        seconds=seconds,
        cache_hit=meta["cache_hit"],
        started_at=started_at,
        completed_at=datetime.now(UTC),
    )


def execute_tool(block: Any, handlers: dict[str, Tool]) -> ToolCall:
    """Run one `tool_use` block that came back from the model.

    `block` is an Anthropic `ToolUseBlock`: it has `id` (the id we must
    echo back on the matching `tool_result`), `name` (which tool the
    model wants), and `input` (the arguments). We look the name up in
    the handler registry and invoke it. Whatever happens, we return a
    `ToolCall` record — never raise — so the runner loop can always
    send *something* back to the model for every tool_use it emitted.

    Two failure modes, both surfaced with `is_error=True`:

    - **unknown tool**: the model asked for a tool we never registered.
      A schema/config mismatch, not a bug in the tool itself.
    - **handler raised**: the tool ran but threw. We stringify the
      exception so the model can read it and decide whether to retry
      with different arguments.

    `dict(block.input)` snapshots the SDK's typed input into a plain
    dict — cheaper to log and store, and decoupled from SDK types.
    """

    t0 = time.perf_counter()
    tool_input = dict(block.input)
    tool = handlers.get(block.name)

    if tool is None:
        output = f"unknown tool: {block.name}"
        is_error = True
    else:
        try:
            output = tool.handler(tool_input)
            is_error = False
        except Exception as e:
            output = f"{type(e).__name__}: {e}"
            is_error = True

    return ToolCall(
        tool_use_id=block.id,
        name=block.name,
        input=tool_input,
        output=output,
        is_error=is_error,
        seconds=time.perf_counter() - t0,
    )
