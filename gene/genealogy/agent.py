"""Genealogy agent: one Conversation scoped to one loaded family.

`build_conversation(family_tag)` opens the SQLite DB for `family_tag` in
read-only mode, reflects its schema into the system prompt, wires the
`run_query` tool from `tools.query`, and returns a `Conversation` ready
to `.ask()`. The DB connection is captured in the tool's closure — no
globals, no per-request opens.

System prompts live as `.md` files under `gene/genealogy/prompts/` with
`{family_tag}` and `{schema}` placeholders. Pass `prompt="<name>"` to
swap variants; the default is `general_conversation`.

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
from gene.agent.prompts import render_prompt
from gene.agent.turn_runner import TurnRunner
from gene.genealogy import PROMPTS_DIR
from gene.genealogy.store import open_db
from gene.genealogy.tools import query as query_tool

def build_system_prompt(
    family_tag: str,
    conn: sqlite3.Connection,
    prompt: str = "general_conversation",
) -> str:
    """Reflect the live schema into the named system-prompt template."""
    return render_prompt(
        prompt,
        PROMPTS_DIR,
        family_tag=family_tag,
        schema=query_tool.describe_schema(conn),
    )


def build_conversation(
    family_tag: str,
    model: str = "sonnet",
    log_path: Path | str | None = None,
    llm: CachedAnthropic | None = None,
    prompt: str = "general_conversation",
    max_steps: int = 20,
) -> Conversation:
    """Assemble a Conversation scoped to the given family DB.

    Opens the SQLite DB read-only, reflects its schema into the system
    prompt, and wires `run_query` as the only tool. `max_steps` caps
    tool-call rounds per turn. Pass a prebuilt `llm` (e.g. from evals) to
    override the default cached client; `model` is ignored in that case.
    `prompt` selects which template under `prompts/` to use.
    """
    conn = open_db(family_tag)
    system = build_system_prompt(family_tag, conn, prompt=prompt)
    if llm is None:
        llm = CachedAnthropic(config=get_llm_config(model=model))
    runner = TurnRunner(llm=llm, tools=[query_tool.make_tool(conn)], max_steps=max_steps)
    return Conversation(runner, system=system, log_path=log_path)
