"""One send-and-execute round trip with the LLM.

`execute_step` is the only place we talk to the model: send the current
messages, and if the response is a `tool_use`, dispatch each requested
tool locally. Returns a `Step` — the observability record for that round.

Free function — takes everything as arguments. Tool machinery (schema
extraction, dispatch, error handling) lives in `gene.agent.tool`; this
module only knows about the send-then-dispatch atom.
"""

import time
from datetime import UTC, datetime
from typing import Any

from gene.agent.llm import CachedAnthropic
from gene.agent.tool import Tool, execute_tools, tool_schemas
from gene.agent.turn import Step


def execute_step(
    llm: CachedAnthropic,
    messages: list[dict[str, Any]],
    system: str | None,
    tools: list[Tool],
) -> Step:
    """Send once, execute any tool_use blocks, return a Step record."""
    started_at = datetime.now(UTC)
    t_step = time.perf_counter()

    t_api = time.perf_counter()
    msg, meta = llm.send(messages=messages, system=system, tools=tool_schemas(tools))
    api_seconds = time.perf_counter() - t_api

    tool_calls = execute_tools(msg, tools)

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
