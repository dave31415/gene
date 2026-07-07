"""Tool types and dispatch — the "everything about tools" module.

Types (`Tool`, `ToolCall`) plus the free functions that translate between
a `list[Tool]` and the Anthropic API's tool-use protocol:

- `tool_schemas(tools)` — the list of schema dicts to pass on the API's
  `tools=` argument. `None` if empty (the API's own convention).
- `find_tool(name, tools)` — linear lookup by schema name.
- `execute_tool(block, tools)` — run one `tool_use` block; always returns
  a `ToolCall`, never raises.
- `execute_tools(response, tools)` — run every `tool_use` block in an
  assistant response. Returns `[]` if the response wasn't asking for tools.
- `tool_result_blocks(tool_calls)` — build the user-message content that
  answers a `tool_use` response on the next request.

Kept in one file because these functions are cheap, tightly coupled, and
only meaningful when read together — separating them would just make the
reader chase files. Callers (`execute_step`, `TurnRunner`) never touch
schemas or handlers directly; they only see `list[Tool]` and the free
functions above.
"""

import time
from collections.abc import Callable
from typing import Any, NamedTuple

from anthropic.types import Message


class Tool(NamedTuple):
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]


class ToolCall(NamedTuple):
    tool_use_id: str
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool
    seconds: float


def tool_schemas(tools: list[Tool]) -> list[dict[str, Any]] | None:
    """The `tools=` payload for the API. `None` when empty."""
    return [t.schema for t in tools] or None


def find_tool(name: str, tools: list[Tool]) -> Tool | None:
    return next((t for t in tools if t.schema["name"] == name), None)


def execute_tool(block: Any, tools: list[Tool]) -> ToolCall:
    """Run one `tool_use` block and return a `ToolCall` record.

    `block` is an Anthropic `ToolUseBlock`: `id` (echoed back on the
    matching `tool_result`), `name` (which tool the model wants), `input`
    (arguments). Two failure modes, both surfaced with `is_error=True`:

    - **unknown tool**: the model asked for a tool we never registered.
      A schema/config mismatch, not a bug in the tool itself.
    - **handler raised**: the tool ran but threw. We stringify the
      exception so the model can read it and decide whether to retry
      with different arguments.

    Never raises — the runner needs *something* to send back for every
    `tool_use` block the model emitted.
    """
    t0 = time.perf_counter()
    tool_input = dict(block.input)
    tool = find_tool(block.name, tools)

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


def execute_tools(response: Message, tools: list[Tool]) -> list[ToolCall]:
    """Dispatch every `tool_use` block in an assistant response.

    Returns `[]` when the response wasn't asking for tools — lets callers
    unconditionally splice the result into a `Step` without a branch.
    """
    if response.stop_reason != "tool_use":
        return []
    return [execute_tool(b, tools) for b in response.content if b.type == "tool_use"]


def tool_result_blocks(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    """Build the user message content that answers a `tool_use` response.

    One `tool_result` block per `tool_use` block from the preceding
    assistant response, paired by `tool_use_id`. Shape dictated by the
    Anthropic Messages API tool-use protocol:

        {
            "type": "tool_result",                      # required literal
            "tool_use_id": str,                         # must match the id
                                                        # from the tool_use block
            "content": str | list[content_block],       # optional; string form
                                                        # is fine for text-only
                                                        # results (list form
                                                        # needed only for images)
            "is_error": bool,                           # optional
        }

    Parallel tool calls in one assistant turn all get their results
    packed into a single user message's content list; ordering doesn't
    matter — the model pairs them by `tool_use_id`.
    """
    return [
        {
            "type": "tool_result",
            "tool_use_id": call.tool_use_id,
            "content": call.output,
            "is_error": call.is_error,
        }
        for call in tool_calls
    ]
