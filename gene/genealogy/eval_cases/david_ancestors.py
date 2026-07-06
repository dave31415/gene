"""Genealogy eval cases against the `david_ancestors` DB.

Direct-ancestor tree only — no uncles, aunts, or cousins. Skipped
cleanly by the runner when the DB isn't loaded locally (private data).

These cases go beyond "did it answer with the right name" and check the
*shape* of the agent's work: did it filter in SQL vs post-hoc, how many
tool-call rounds it took. That's the point of having the Turn-level
observability.

This module owns its genealogy-specific glue — `build_conversation(llm)`
and `precheck()` — so the core runner in `gene.agent.evals` needs zero
knowledge of the genealogy package.
"""

from gene.agent.eval_case import TurnCase
from gene.agent.eval_predicates import contains_all
from gene.agent.eval_predicates_turn import max_steps, sql_matches, used_tool
from gene.agent.llm import CachedAnthropic
from gene.genealogy.agent import build_conversation as _build
from gene.genealogy.config import get_db_path

FAMILY_TAG = "david_ancestors"


def build_conversation(llm: CachedAnthropic):
    """Runner-facing factory: hand back a Conversation scoped to this suite's DB."""
    return _build(FAMILY_TAG, llm=llm)


def precheck() -> str | None:
    """Return a skip reason when the family DB isn't built; None otherwise."""
    if not get_db_path(FAMILY_TAG).exists():
        return f"data '{FAMILY_TAG}' not built (run: python -m gene.genealogy.load {FAMILY_TAG})"
    return None


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
    TurnCase(
        name="ancestor_walk_uses_recursive_cte",
        # Observed head-to-head: Opus solves this in 3 queries with a
        # WITH RECURSIVE walking the ancestor chain, then answers. Haiku
        # walks one hop at a time, hits max_steps at 20, and produces no
        # reply. This case fails any model that doesn't reach for
        # recursive SQL on an ancestor question — the whole point of
        # TurnCase, since a one-shot text check would miss it.
        prompt="Is David Johnston related to John Rice?",
        check=lambda t: (
            contains_all(t.text, ["Rice"])
            and sql_matches(t, r"with\s+recursive\b")
            and max_steps(t, 8)
        ),
    ),
]
