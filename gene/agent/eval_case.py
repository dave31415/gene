"""Types for eval case suites.

Case suites live in `gene.agent.eval_cases.*` (core) and
`gene.genealogy.eval_cases.*` (genealogy) and depend only on this module —
never on the runner in `gene.agent.evals`. Keeping suites as pure data avoids
circular imports and lets tooling load a suite without pulling in the
Anthropic client.

Two case flavours:
- `Case` — one-shot: check receives the raw `Message` from `llm.send`.
- `TurnCase` — full agent loop: check receives a `Turn` (see gene.agent.turn)
  so it can inspect tool calls, step count, tokens, and final text. Named
  `TurnCase` (not `AgentCase`) to leave room for `MultiTurnCase` or
  `ScriptCase` later without ambiguity about scope.

A suite module exposes `CASES: list[Case | TurnCase]` and, for turn-based
suites that need a DB, `TAG: str` naming the family DB. Suites with a TAG
are skipped cleanly when the DB isn't loaded.
"""

from collections.abc import Callable
from typing import NamedTuple

from anthropic.types import Message

from gene.agent.turn import Turn


class Case(NamedTuple):
    name: str
    prompt: str
    check: Callable[[Message], bool]


class TurnCase(NamedTuple):
    name: str
    prompt: str
    check: Callable[[Turn], bool]


class Result(NamedTuple):
    case: Case | TurnCase
    passed: bool
    input_tokens: int
    output_tokens: int
    seconds: float
    steps: int          # 1 for one-shot; ≥1 for turn-based
    tool_calls: int     # 0 for one-shot; ≥0 for turn-based


class Report(NamedTuple):
    model: str
    results: list[Result]
    passed: int
    total: int
    total_input_tokens: int
    total_output_tokens: int
    total_seconds: float


def text(msg: Message) -> str:
    """Concatenate all text blocks from a response."""
    return "".join(b.text for b in msg.content if b.type == "text")
