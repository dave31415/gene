import time
import uuid
from datetime import UTC, datetime
from typing import Any

from anthropic.types import Message

from gene.llm import CachedAnthropic
from gene.tool import Tool, ToolCall
from gene.turn import Step, Turn, TurnError


def tool_result_blocks(step: Step) -> list[dict[str, Any]]:
    """Build the user message content that answers a tool_use response.

    The returned list is what goes into the *next* user message's `content`
    field — one `tool_result` block per `tool_use` block from the preceding
    assistant response, paired by `tool_use_id`. The shape is dictated by
    the Anthropic Messages API tool-use protocol, not by us:

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
        for call in step.tool_calls
    ]


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
        """Run one turn: send → tool_use → send → ... → end_turn.

        The loop accumulates two parallel views of what happened:

        - `steps`: rich per-round record (request sent, response, tool
          calls executed, timings, cache hit). Observability substrate.
        - `new_messages`: the flat API view — the initial user prompt,
          every assistant response, and every tool_result we sent back.
          The *next* iteration's request is `messages + new_messages`,
          and the caller splices this into their history after we return.

        `new_messages` is losslessly derivable from `steps + user_input`,
        but we build both eagerly because the loop already needs
        `new_messages` in hand to construct each request.

        Three scalars are decided at loop exit:

        - `terminal_reason`: why we stopped ("end_turn", "error",
          "max_steps", or another stop_reason from the model).
        - `error`: a `TurnError` if an exception was caught, else None.
        - `final_message`: the terminal (non-tool_use) response, or None
          if we exited via error or max_steps.

        `messages` is not mutated. Exceptions raised inside a step are
        caught and recorded on `Turn.error` — the caller always receives
        a Turn back with whatever partial state was accumulated.
        """
        # --- set once at turn start ---
        turn_id = uuid.uuid4().hex
        started_at = datetime.now(UTC)

        # --- accumulated across steps ---
        steps: list[Step] = []
        new_messages: list[dict[str, Any]] = [{"role": "user", "content": user_input}]

        # --- decided at loop exit; defaults cover the "hit max_steps" case ---
        terminal_reason: str = "max_steps"
        error: TurnError | None = None
        final_message: Message | None = None

        for i in range(self.max_steps):
            # Any failure inside the step — network, API error, unexpected bug —
            # becomes a TurnError. We return whatever partial state we have.
            try:
                step = self._one_step(messages + new_messages, system)
            except Exception as e:
                error = TurnError(type=type(e).__name__, message=str(e), step_index=i)
                terminal_reason = "error"
                break

            steps.append(step)
            # Mirror the assistant response (including any tool_use blocks)
            # into the API view so the next request stays consistent.
            # Convert content blocks to plain dicts via `model_dump(mode="json")`
            # so `new_messages` is fully JSON-native — no Pydantic types leak
            # into the history the caller splices back in.
            content = [b.model_dump(mode="json") for b in step.response.content]
            new_messages.append({"role": "assistant", "content": content})

            # Non-tool_use stop reason means the model is done talking this turn.
            if step.response.stop_reason != "tool_use":
                terminal_reason = str(step.response.stop_reason or "unknown")
                final_message = step.response
                break

            # Model asked to use tools; the tools already ran inside _one_step.
            # Feed the results back as a user message and loop for another send.
            new_messages.append({"role": "user", "content": tool_result_blocks(step)})

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
            tool_calls = [self._execute_tool(b) for b in msg.content if b.type == "tool_use"]

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

    def _execute_tool(self, block: Any) -> ToolCall:
        """Run one `tool_use` block that came back from the model.

        `block` is an Anthropic `ToolUseBlock`: it has `id` (the id we must
        echo back on the matching `tool_result`), `name` (which tool the
        model wants), and `input` (the arguments). We look the name up in
        our handler registry and invoke it. Whatever happens, we return a
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
        tool = self._handlers.get(block.name)

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
