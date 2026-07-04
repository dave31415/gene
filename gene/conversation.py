"""Stateful chat session on top of a `TurnRunner`.

`Conversation` owns the message history (the flat list the API needs)
and a parallel list of `Turn` objects (the structured view for
observability and evals). Every `ask()` delegates the actual work —
including tool loops — to the runner, then splices the resulting
`turn.new_messages` into history.

No persistence — history disappears when the object is garbage-collected.
"""

from gene.turn import Turn, TurnRunner


class Conversation:
    """One chat session. Stateful. Owns history; delegates each turn to a `TurnRunner`."""

    def __init__(self, runner: TurnRunner, system: str | None = None):
        self.runner = runner
        self.system = system
        self.history: list[dict] = []
        self.turns: list[Turn] = []

    def ask(self, text: str) -> Turn:
        """Run one turn: send, execute any tools, loop until end_turn."""
        turn = self.runner.run(self.history, text, system=self.system)
        self.history.extend(turn.new_messages)
        self.turns.append(turn)
        return turn
