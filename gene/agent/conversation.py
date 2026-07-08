"""Stateful chat session on top of a `TurnRunner`.

`Conversation` owns the message history (the flat list the API needs)
and a parallel list of `Turn` objects (the structured view for
observability and evals). It also owns the *semantics* the runner is
told to use each turn — the system prompt, the tools available, and
the per-turn step cap. Every `ask()` passes these to the runner along
with the current history; the runner does the send/tool loop and hands
back a Turn, which we splice into history.

Before splicing, `create_turn_memory` decides what actually gets
remembered: user + assistant text stays, tool_use / tool_result
blocks get dropped. The model rarely consults prior-turn tool traffic
and re-shipping it dominates the context budget — see
`gene.agent.log_view --budget`.

If `log_path` is given, each completed Turn is appended as one JSON
line, giving a crash-safe, grep-friendly record across sessions.
"""

import json
from pathlib import Path
from typing import Any

from gene.agent.tool import Tool
from gene.agent.turn import Turn
from gene.agent.turn_runner import TurnRunner


def _block_type(block: Any) -> str:
    if isinstance(block, dict):
        return block.get("type", "")
    return getattr(block, "type", "")


def create_turn_memory(new_messages: list[dict]) -> list[dict]:
    """Return the subset of a completed turn's messages that should persist.

    Keeps user text and assistant text blocks; drops tool_use and
    tool_result blocks. Messages that end up empty after stripping are
    themselves dropped so the resulting list stays API-valid.
    """
    kept: list[dict] = []
    for msg in new_messages:
        content = msg["content"]
        if isinstance(content, str):
            kept.append(msg)
            continue
        text_blocks = [b for b in content if _block_type(b) == "text"]
        if text_blocks:
            kept.append({"role": msg["role"], "content": text_blocks})
    return kept


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

        self.history.extend(create_turn_memory(turn.new_messages))
        self.turns.append(turn)

        if self.log_path is not None:
            with self.log_path.open("a") as f:
                f.write(json.dumps(turn.to_dict()) + "\n")
        return turn
