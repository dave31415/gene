"""The turn loop: send → tool_use → send → ... → end_turn.

`TurnRunner` holds the one thing that stays constant across every turn
of every composer: the LLM client. Everything else — the system prompt,
the tools available on this turn, the step cap — is passed to `run()`
per call, because different composers (a plain chatbot, a plan/execute
agent, a mode-switching state machine) use different values on
different turns.

The runner knows about turn shape (steps, new_messages, terminal
reasons) and delegates the send-and-dispatch atom to `execute_step`
and the tool-result formatting to `tool.tool_result_blocks`. It never
touches schemas, handlers, or `tool_use` blocks directly.

Exceptions raised during a step (API errors, unexpected failures) are
caught and recorded on `Turn.error`; the caller always receives a Turn.
`KeyboardInterrupt` and `SystemExit` propagate as usual.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from anthropic.types import Message

from gene.agent.execute_step import execute_step
from gene.agent.llm import CachedAnthropic
from gene.agent.tool import Tool, tool_result_blocks
from gene.agent.turn import Step, Turn, TurnError


class TurnRunner:
    """Executes one turn: the send → tool_use → send → ... → end_turn loop.

    Holds only the LLM client — the one invariant across every turn of
    every composer. `system`, `tools`, and `max_steps` are per-call
    because composers routinely vary them turn-to-turn.
    """

    def __init__(self, llm: CachedAnthropic):
        self.llm = llm

    def run(
        self,
        messages: list[dict[str, Any]],
        user_input: str,
        system: str | None = None,
        tools: list[Tool] | None = None,
        max_steps: int = 10,
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
        tools = tools or []

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

        for i in range(max_steps):
            # Any failure inside the step — network, API error, unexpected bug —
            # becomes a TurnError. We return whatever partial state we have.
            try:
                step = execute_step(
                    self.llm,
                    messages + new_messages,
                    system,
                    tools,
                )
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

            # Model asked to use tools; the tools already ran inside execute_step.
            # Feed the results back as a user message and loop for another send.
            new_messages.append(
                {"role": "user", "content": tool_result_blocks(step.tool_calls)}
            )

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
