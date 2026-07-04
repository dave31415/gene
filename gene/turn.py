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

from datetime import datetime
from typing import Any, NamedTuple

from anthropic.types import Message

from gene.tool import ToolCall


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
                lines.append(f"    {mark}{tc.name}({tc.input}) → {tc.output}  [{tc.tool_use_id}]")
        lines.append(f"  terminal: {self.terminal_reason}")
        if self.error is not None:
            lines.append(
                f"  error: {self.error.type}: {self.error.message} "
                f"(step_index={self.error.step_index})"
            )
        return "\n".join(lines)
