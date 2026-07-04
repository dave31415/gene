"""Stateful chat session on top of a `TurnRunner`.

`Conversation` owns the message history (the flat list the API needs)
and a parallel list of `Turn` objects (the structured view for
observability and evals). Every `ask()` delegates the actual work —
including tool loops — to the runner, then splices the resulting
`turn.new_messages` into history.

If `log_path` is given, each completed Turn is appended as one JSON
line, giving a crash-safe, grep-friendly record across sessions.
"""

import json
from pathlib import Path

from gene.turn import Turn
from gene.turn_runner import TurnRunner


class Conversation:
    """One chat session. Stateful. Owns history; delegates each turn to a `TurnRunner`."""

    def __init__(
        self,
        runner: TurnRunner,
        system: str | None = None,
        log_path: Path | str | None = None,
    ):
        self.runner = runner
        self.system = system
        self.history: list[dict] = []
        self.turns: list[Turn] = []
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def ask(self, text: str) -> Turn:
        """Run one turn: send, execute any tools, loop until end_turn."""
        turn = self.runner.run(self.history, text, system=self.system)
        self.history.extend(turn.new_messages)
        self.turns.append(turn)
        if self.log_path is not None:
            with self.log_path.open("a") as f:
                f.write(json.dumps(turn.to_dict()) + "\n")
        return turn
