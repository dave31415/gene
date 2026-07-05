"""Genealogy agent: one Conversation scoped to one loaded family.

`build_conversation(tag)` opens the SQLite DB for `tag` in read-only mode,
reflects its schema into the system prompt, wires the `run_query` tool from
`tools.query`, and returns a `Conversation` ready to `.ask()`. The DB
connection is captured in the tool's closure — no globals, no per-request
opens.

The tool interface is deliberately narrow: a single `sql` string in,
`{"rows", "row_count", "truncated"}` (or `{"error"}`) JSON string out.
Higher-level graph-walk tools (ancestors, descendants, common ancestor)
are held back until we see where SQL is clumsy in practice.
"""

import sqlite3
from pathlib import Path

from gene.agent.config import get_llm_config
from gene.agent.conversation import Conversation
from gene.agent.llm import CachedAnthropic
from gene.agent.turn_runner import TurnRunner
from gene.genealogy.store import open_db
from gene.genealogy.tools import query as query_tool

_MAX_STEPS = 20

_SYSTEM_TEMPLATE = """\
You are a genealogy assistant answering questions about the "{tag}" family tree.
All data lives in a read-only SQLite database with the schema below. Use the
run_query tool to answer questions — do not guess from prior knowledge.

Schema:
{schema}

Conventions:
- Individual and family ids are GEDCOM xrefs like "@I1@", "@F1@".
- full_name preserves the source form with the surname in slashes, e.g.
  "David /Johnston/". given/surname are the parsed parts.
- date_year is the extracted 4-digit year (int, nullable). date_raw is the
  source date string (nullable). Filter on date_year for ranges; show
  date_raw when quoting a date to the user.
- Parent links go through the families table: an individual's parents are
  the husband_id/wife_id of the family whose id appears as family_id in
  family_children for that individual.
- If a query returns zero rows, say so — do not invent an answer.
- Prefer specific SELECT lists over SELECT *.
- Reply in plain text only — no markdown, no bold, no tables, no bullet
  characters. Answers are read in a terminal.
"""


def build_system_prompt(tag: str, conn: sqlite3.Connection) -> str:
    """Reflect the live schema into the system prompt."""
    return _SYSTEM_TEMPLATE.format(tag=tag, schema=query_tool.describe_schema(conn))


def build_conversation(
    tag: str,
    model: str = "sonnet",
    log_path: Path | str | None = None,
) -> Conversation:
    """Assemble a Conversation scoped to the given family DB.

    Opens the SQLite DB read-only, reflects its schema into the system
    prompt, and wires `run_query` as the only tool. `max_steps=20` caps
    tool-call rounds per turn.
    """
    conn = open_db(tag)
    system = build_system_prompt(tag, conn)
    llm = CachedAnthropic(config=get_llm_config(model=model))
    runner = TurnRunner(llm=llm, tools=[query_tool.make_tool(conn)], max_steps=_MAX_STEPS)
    return Conversation(runner, system=system, log_path=log_path)
