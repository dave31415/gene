"""Stateful chat session on top of a `TurnRunner`.

`Conversation` owns the message history (the flat list the API needs)
and a parallel list of `Turn` objects (the structured view for
observability and evals). It also owns the *semantics* the runner is
told to use each turn — the system prompt, the tools available, and
the per-turn step cap. Every `ask()` passes these to the runner along
with the current history; the runner does the send/tool loop and hands
back a Turn, which we splice into history.

If `log_path` is given, each completed Turn is appended as one JSON
line, giving a crash-safe, grep-friendly record across sessions.
"""

import json
from pathlib import Path

from gene.agent.tool import Tool
from gene.agent.turn import Turn
from gene.agent.turn_runner import TurnRunner


class Conversation:
    """One chat session. Stateful. Owns history + semantics; delegates each turn to a `TurnRunner`."""

    def __init__(
        self,
        runner: TurnRunner,
        system: str | None = None,
        tools: list[Tool] | None = None,
        max_steps: int = 10,
        log_path: Path | str | None = None,
    ):
        self.runner = runner
        self.system = system
        self.tools = tools or []
        self.max_steps = max_steps
        self.history: list[dict] = []
        self.turns: list[Turn] = []
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def ask(self, text: str) -> Turn:
        """Run one turn: send, execute any tools, loop until end_turn."""
        turn = self.runner.run(
            self.history,
            text,
            system=self.system,
            tools=self.tools,
            max_steps=self.max_steps,
        )

        # The following two lines are where context is managed
        # Simply append new messages for now
        # eventually or in other classes, manage context more generally

        self.history.extend(turn.new_messages)
        self.turns.append(turn)

        if self.log_path is not None:
            with self.log_path.open("a") as f:
                f.write(json.dumps(turn.to_dict()) + "\n")
        return turn
