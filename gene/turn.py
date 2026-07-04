"""One user prompt → one final assistant response, possibly with tool calls.

A `Turn` is the top-level unit of interaction. It contains N `Step`s
(one per API round-trip). A step with `stop_reason == "tool_use"`
triggers tool execution and another step; the loop ends on `end_turn`
or when a safety cap is hit.

`TurnRunner` executes turns. It's stateless per call — callers pass
the current message history in and receive a `Turn` back. The
history-mutating `Conversation` class is a separate shell on top.
"""

import time
from typing import Any, NamedTuple

from anthropic.types import Message

from gene.llm import CachedAnthropic
from gene.tool import Tool, ToolCall


class Step(NamedTuple):
    """One API round-trip inside a turn."""

    request: dict[str, Any]
    response: Message
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    seconds: float
    cache_hit: bool


class Turn(NamedTuple):
    """One user prompt → one final assistant response."""

    user_input: str
    steps: list[Step]
    new_messages: list[dict[str, Any]]
    final_message: Message
    terminal_reason: str  # "end_turn" | "max_steps" | "stop_sequence" | "max_tokens" | ...

    @property
    def text(self) -> str:
        """Join every text block in the final assistant message."""
        return "".join(b.text for b in self.final_message.content if b.type == "text")

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.steps)

    def summary(self) -> str:
        """One-line human summary. For chat_loop debugging or logs."""
        tool_names = [tc.name for s in self.steps for tc in s.tool_calls]
        tools_str = f", tools={tool_names}" if tool_names else ""
        cache_hits = sum(1 for s in self.steps if s.cache_hit)
        return (
            f"steps={len(self.steps)}{tools_str} | "
            f"tokens={self.input_tokens} in / {self.output_tokens} out | "
            f"{self.seconds:.2f}s | cache {cache_hits}/{len(self.steps)} hit | "
            f"reason={self.terminal_reason}"
        )

    def trace(self) -> str:
        """Multi-line per-step dump. For verbose debugging."""
        lines = [f"Turn: {self.user_input!r}"]
        for i, s in enumerate(self.steps):
            hit = "hit" if s.cache_hit else "miss"
            lines.append(
                f"  step {i}: {s.response.stop_reason} | "
                f"{s.input_tokens} in / {s.output_tokens} out | "
                f"{s.seconds:.2f}s | cache {hit}"
            )
            for tc in s.tool_calls:
                mark = "ERR " if tc.is_error else "    "
                lines.append(f"    {mark}{tc.name}({tc.input}) → {tc.output}")
        lines.append(f"  terminal: {self.terminal_reason}")
        return "\n".join(lines)


class TurnRunner:
    """Executes one turn: the send → tool_use → send → ... → end_turn loop.

    Stateless per `run()`. Holds the pieces that stay constant across turns:
    the LLM client, the tool registry, the step cap. History is passed in
    each call — no state accumulates on the runner itself.
    """

    def __init__(
        self,
        llm: CachedAnthropic,
        tools: list[Tool] | None = None,
        max_steps: int = 10,
    ):
        self.llm = llm
        self.max_steps = max_steps
        self._schemas: list[dict[str, Any]] = [t.schema for t in (tools or [])]
        self._handlers: dict[str, Tool] = {t.schema["name"]: t for t in (tools or [])}

    def run(
        self,
        messages: list[dict[str, Any]],
        user_input: str,
        system: str | None = None,
    ) -> Turn:
        """Run one turn. `messages` is the prior history (not mutated).
        Returns a Turn whose `new_messages` should be appended to the caller's
        history to keep the API view consistent."""
        new_messages: list[dict[str, Any]] = [{"role": "user", "content": user_input}]
        steps: list[Step] = []
        terminal_reason = "max_steps"

        for _ in range(self.max_steps):
            request_messages = messages + new_messages
            step = self._one_step(request_messages, system)
            steps.append(step)
            new_messages.append({"role": "assistant", "content": step.response.content})

            if step.response.stop_reason != "tool_use":
                terminal_reason = step.response.stop_reason or "unknown"
                break

            new_messages.append({"role": "user", "content": self._tool_result_blocks(step)})

        return Turn(
            user_input=user_input,
            steps=steps,
            new_messages=new_messages,
            final_message=steps[-1].response,
            terminal_reason=terminal_reason,
        )

    def _one_step(self, messages: list[dict[str, Any]], system: str | None) -> Step:
        """Send once, execute any tool_use blocks, return a Step record."""
        tools_arg = self._schemas if self._schemas else None
        t0 = time.perf_counter()
        msg, meta = self.llm.send(messages=messages, system=system, tools=tools_arg)
        # request is what build_request would have produced — reconstruct minimally
        # for observability. The exact same dict was hashed for the cache key.
        request = {"messages": messages, "system": system, "tools": tools_arg}
        tool_calls: list[ToolCall] = []
        if msg.stop_reason == "tool_use":
            tool_calls = [self._execute(b) for b in msg.content if b.type == "tool_use"]
        elapsed = time.perf_counter() - t0
        return Step(
            request=request,
            response=msg,
            tool_calls=tool_calls,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            seconds=elapsed,
            cache_hit=meta["cache_hit"],
        )

    def _execute(self, block: Any) -> ToolCall:
        """Run one tool_use block through the handler registry."""
        t0 = time.perf_counter()
        tool = self._handlers.get(block.name)
        if tool is None:
            return ToolCall(
                name=block.name,
                input=dict(block.input),
                output=f"unknown tool: {block.name}",
                is_error=True,
                seconds=time.perf_counter() - t0,
            )
        try:
            output = tool.handler(dict(block.input))
            is_error = False
        except Exception as e:
            output = f"{type(e).__name__}: {e}"
            is_error = True
        return ToolCall(
            name=block.name,
            input=dict(block.input),
            output=output,
            is_error=is_error,
            seconds=time.perf_counter() - t0,
        )

    def _tool_result_blocks(self, step: Step) -> list[dict[str, Any]]:
        """Build the user message content that answers a tool_use response.
        One tool_result block per tool_use block in the assistant message,
        referenced by tool_use_id."""
        ids = [b.id for b in step.response.content if b.type == "tool_use"]
        return [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": call.output,
                "is_error": call.is_error,
            }
            for tool_use_id, call in zip(ids, step.tool_calls, strict=True)
        ]
