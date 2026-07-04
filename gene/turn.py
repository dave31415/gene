"""One user prompt → one final assistant response, possibly with tool calls.

A `Turn` is the top-level unit of interaction. It contains N `Step`s
(one per API round-trip). A step with `stop_reason == "tool_use"`
triggers tool execution and another step; the loop ends on `end_turn`,
a safety cap, or an unrecoverable error.

`TurnRunner` executes turns. It's stateless per call — callers pass
the current message history in and receive a `Turn` back. The
history-mutating `Conversation` class is a separate shell on top.

Every level records enough for post-hoc inspection: full request as
sent, cache-hit info, split API-vs-total timing, absolute timestamps,
per-tool inputs/outputs. Exceptions are caught and surfaced as
`Turn.error`; the partial Turn is still returned so no observability
is lost when things break.
"""

import time
import uuid
from datetime import UTC, datetime
from typing import Any, NamedTuple

from anthropic.types import Message

from gene.llm import CachedAnthropic
from gene.tool import Tool, ToolCall


class TurnError(NamedTuple):
    """Recorded when the turn loop terminated because of an exception."""

    type: str
    message: str
    step_index: int | None  # which step was in flight; None if pre-loop


class Step(NamedTuple):
    """One API round-trip inside a turn."""

    request: dict[str, Any]  # exact dict sent to the Anthropic API
    response: Message
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    api_seconds: float  # send() wall clock only
    seconds: float  # total: send + tool executions this step
    cache_hit: bool
    started_at: datetime
    completed_at: datetime


class Turn(NamedTuple):
    """One user prompt → one final assistant response."""

    id: str  # uuid4 hex; handle for logs and (later) an introspection tool
    user_input: str
    steps: list[Step]
    new_messages: list[dict[str, Any]]
    final_message: Message | None  # None if the turn errored before end_turn
    terminal_reason: str  # "end_turn" | "max_steps" | "error" | "max_tokens" | ...
    error: TurnError | None
    started_at: datetime
    completed_at: datetime

    @property
    def text(self) -> str:
        """Join every text block in the final assistant message. Empty on error."""
        if self.final_message is None:
            return ""
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

    @property
    def api_seconds(self) -> float:
        return sum(s.api_seconds for s in self.steps)

    def summary(self) -> str:
        """One-line human summary. For chat_loop debugging or logs."""
        tool_names = [tc.name for s in self.steps for tc in s.tool_calls]
        tools_str = f", tools={tool_names}" if tool_names else ""
        cache_hits = sum(1 for s in self.steps if s.cache_hit)
        reason = self.terminal_reason
        if self.error is not None:
            reason = f"error ({self.error.type}: {self.error.message})"
        return (
            f"id={self.id[:8]} steps={len(self.steps)}{tools_str} | "
            f"tokens={self.input_tokens} in / {self.output_tokens} out | "
            f"{self.seconds:.2f}s ({self.api_seconds:.2f}s api) | "
            f"cache {cache_hits}/{len(self.steps)} hit | reason={reason}"
        )

    def trace(self) -> str:
        """Multi-line per-step dump. For verbose debugging."""
        lines = [
            f"Turn {self.id[:8]}: {self.user_input!r}",
            f"  started: {self.started_at.isoformat()}",
            f"  completed: {self.completed_at.isoformat()}",
        ]
        for i, s in enumerate(self.steps):
            hit = "hit" if s.cache_hit else "miss"
            lines.append(
                f"  step {i}: {s.response.stop_reason} | "
                f"{s.input_tokens} in / {s.output_tokens} out | "
                f"{s.seconds:.2f}s total ({s.api_seconds:.2f}s api) | cache {hit}"
            )
            for tc in s.tool_calls:
                mark = "ERR " if tc.is_error else "    "
                lines.append(
                    f"    {mark}{tc.name}({tc.input}) → {tc.output}  [{tc.tool_use_id}]"
                )
        lines.append(f"  terminal: {self.terminal_reason}")
        if self.error is not None:
            lines.append(
                f"  error: {self.error.type}: {self.error.message} "
                f"(step_index={self.error.step_index})"
            )
        return "\n".join(lines)


class TurnRunner:
    """Executes one turn: the send → tool_use → send → ... → end_turn loop.

    Stateless per `run()`. Holds the pieces that stay constant across turns:
    the LLM client, the tool registry, the step cap. History is passed in
    each call — no state accumulates on the runner itself.

    Exceptions raised during a step (API errors, unexpected failures) are
    caught and recorded on `Turn.error`; the caller always receives a Turn.
    `KeyboardInterrupt` and `SystemExit` propagate as usual.
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
        turn_id = uuid.uuid4().hex
        started_at = datetime.now(UTC)
        new_messages: list[dict[str, Any]] = [{"role": "user", "content": user_input}]
        steps: list[Step] = []
        terminal_reason: str = "max_steps"
        error: TurnError | None = None
        final_message: Message | None = None

        for i in range(self.max_steps):
            try:
                request_messages = messages + new_messages
                step = self._one_step(request_messages, system)
            except Exception as e:
                error = TurnError(type=type(e).__name__, message=str(e), step_index=i)
                terminal_reason = "error"
                break

            steps.append(step)
            new_messages.append({"role": "assistant", "content": step.response.content})

            if step.response.stop_reason != "tool_use":
                terminal_reason = step.response.stop_reason or "unknown"
                final_message = step.response
                break

            new_messages.append({"role": "user", "content": self._tool_result_blocks(step)})

        return Turn(
            id=turn_id,
            user_input=user_input,
            steps=steps,
            new_messages=new_messages,
            final_message=final_message,
            terminal_reason=terminal_reason,
            error=error,
            started_at=started_at,
            completed_at=datetime.now(UTC),
        )

    def _one_step(self, messages: list[dict[str, Any]], system: str | None) -> Step:
        """Send once, execute any tool_use blocks, return a Step record."""
        tools_arg = self._schemas if self._schemas else None
        started_at = datetime.now(UTC)
        t_step = time.perf_counter()

        t_api = time.perf_counter()
        msg, meta = self.llm.send(messages=messages, system=system, tools=tools_arg)
        api_seconds = time.perf_counter() - t_api

        tool_calls: list[ToolCall] = []
        if msg.stop_reason == "tool_use":
            tool_calls = [self._execute(b) for b in msg.content if b.type == "tool_use"]

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

    def _execute(self, block: Any) -> ToolCall:
        """Run one tool_use block through the handler registry."""
        t0 = time.perf_counter()
        tool = self._handlers.get(block.name)
        if tool is None:
            return ToolCall(
                tool_use_id=block.id,
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
            tool_use_id=block.id,
            name=block.name,
            input=dict(block.input),
            output=output,
            is_error=is_error,
            seconds=time.perf_counter() - t0,
        )

    def _tool_result_blocks(self, step: Step) -> list[dict[str, Any]]:
        """Build the user message content that answers a tool_use response.
        One tool_result block per tool_use, referenced by tool_use_id."""
        return [
            {
                "type": "tool_result",
                "tool_use_id": call.tool_use_id,
                "content": call.output,
                "is_error": call.is_error,
            }
            for call in step.tool_calls
        ]
