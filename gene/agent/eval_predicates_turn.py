"""Predicates that inspect a `Turn` — step count, tool calls, SQL shape.

Kept separate from `eval_predicates` (text-only) so one-shot suites don't
have to import `Turn` just to use `contains_all`. Combine both freely in
a `TurnCase.check`:

    check=lambda t: (
        contains_all(t.text, ["John Dennis"])
        and max_steps(t, 4)
        and sql_matches(t, r"\\bsex\\s*=")
    )
"""

import re

from gene.agent.turn import Turn


def max_steps(turn: Turn, n: int) -> bool:
    """Cap on how many API round-trips the turn took."""
    return len(turn.steps) <= n


def used_tool(turn: Turn, name: str) -> bool:
    """True iff the turn called the named tool at least once."""
    return any(tc.name == name for s in turn.steps for tc in s.tool_calls)


def sql_matches(turn: Turn, pattern: str) -> bool:
    """True iff any run_query call's `sql` input matches the regex."""
    rx = re.compile(pattern, re.IGNORECASE)
    for s in turn.steps:
        for tc in s.tool_calls:
            sql = tc.input.get("sql", "") if isinstance(tc.input, dict) else ""
            if rx.search(sql):
                return True
    return False
