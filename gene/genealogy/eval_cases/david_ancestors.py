"""Genealogy eval cases against the `david_ancestors` DB.

Direct-ancestor tree only — no uncles, aunts, or cousins. Skipped
cleanly by the runner when the DB isn't loaded locally (private data).

These cases go beyond "did it answer with the right name" and check the
*shape* of the agent's work: did it filter in SQL vs post-hoc, how many
tool-call rounds it took. That's the point of having the Turn-level
observability.
"""

from gene.agent.eval_case import TurnCase
from gene.agent.eval_predicates import contains_all
from gene.agent.eval_predicates_turn import max_steps, sql_matches, used_tool

TAG = "david_ancestors"

CASES: list[TurnCase] = [
    TurnCase(
        name="davids_parents",
        prompt="Who are David Johnston's parents? Answer with just their full names.",
        check=lambda t: (
            contains_all(t.text, ["John Dennis", "Mary Catherine"])
            and used_tool(t, "run_query")
            and max_steps(t, 4)
        ),
    ),
    TurnCase(
        name="paternal_grandfather_surname_in_sql",
        # Ancestors-only DB, so no siblings — the interesting behaviour is
        # whether it filters in SQL. We check the answer AND that at least
        # one query used a WHERE clause narrowing by surname or given name.
        prompt="Who is David Johnston's paternal grandfather? Answer with just the full name.",
        check=lambda t: (
            "Johnston" in t.text
            and sql_matches(t, r"where\b.*\b(surname|given|full_name|id)\b")
            and max_steps(t, 6)
        ),
    ),
]
