"""Types and helpers for eval case suites.

Case suites live in `gene.agent.eval_cases.*` and depend only on this module —
never on the runner in `gene.agent.evals`. Keeping suites as pure data avoids
circular imports and lets tooling load a suite without pulling in the
Anthropic client.
"""

from collections.abc import Callable
from typing import NamedTuple

from anthropic.types import Message


class Case(NamedTuple):
    name: str
    prompt: str
    check: Callable[[Message], bool]


class Result(NamedTuple):
    case: Case
    passed: bool
    input_tokens: int
    output_tokens: int
    seconds: float


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
