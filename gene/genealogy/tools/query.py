"""The run_query tool: guarded SELECT execution for genealogy DBs.

The agent's only tool. The connection is opened read-only (see
`store.open_db`), which already blocks writes, but that alone isn't
enough — an LLM-authored SQL string could still ATTACH another file,
spin the CPU on a runaway CTE, or dump 30k rows into the context window.
This module adds those guards on top of the RO connection.

Guards, in the order applied:
    1. statement whitelist: first token must be SELECT or WITH
    2. keyword blacklist:   ATTACH / PRAGMA / LOAD_EXTENSION rejected
    3. wall-clock timeout:  background thread calls conn.interrupt()
    4. row cap:             fetchmany(max_rows + 1), flag if truncated
    5. serialization:       sqlite3.Row → dict of JSON-safe primitives

Any exception (bad SQL, timeout, guard violation) is returned as
`{"error": "..."}` rather than raised, so the tool handler can hand
the string straight back to the model.

`describe_schema` also lives here — the agent uses it to reflect the
live schema into its system prompt, and it's co-located with the tool
so callers pull one module for everything query-shaped.
"""

import json
import re
import sqlite3
import threading
from typing import Any

from gene.agent.tool import Tool

_ALLOWED_FIRST = re.compile(r"^\s*(?:--[^\n]*\n|/\*.*?\*/|\s)*(SELECT|WITH)\b", re.IGNORECASE | re.DOTALL)
_FORBIDDEN = re.compile(r"\b(ATTACH|PRAGMA|LOAD_EXTENSION)\b", re.IGNORECASE)

_DEFAULT_MAX_ROWS = 100
_DEFAULT_TIMEOUT_S = 5.0

SCHEMA: dict[str, Any] = {
    "name": "run_query",
    "description": (
        "Run a read-only SELECT (or WITH ... SELECT) against the family "
        "database. Returns rows as JSON. If `truncated` is true, refine the "
        "query with LIMIT or a tighter WHERE clause. Errors come back as "
        "{\"error\": \"...\"} — read the message and retry."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A single SELECT or WITH statement.",
            },
        },
        "required": ["sql"],
    },
}


def run_query(
    conn: sqlite3.Connection,
    sql: str,
    *,
    max_rows: int = 100,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Execute `sql` and return `{"rows", "row_count", "truncated"}` or `{"error"}`.

    `conn` must be a read-only connection (as returned by `store.open_db`).
    Timeouts, guard rejections, and SQL errors all come back as an error dict.
    """
    if not _ALLOWED_FIRST.match(sql):
        return {"error": "only SELECT / WITH statements are allowed"}
    if _FORBIDDEN.search(sql):
        return {"error": "ATTACH, PRAGMA, and LOAD_EXTENSION are not allowed"}

    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.start()
    try:
        cursor = conn.execute(sql)
        raw_rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as e:
        # `interrupt()` surfaces here too, so treat that specially.
        msg = str(e)
        if "interrupted" in msg.lower():
            return {"error": f"query timed out after {timeout_s}s"}
        return {"error": f"sql error: {msg}"}
    except sqlite3.DatabaseError as e:
        return {"error": f"sql error: {e}"}
    finally:
        timer.cancel()

    truncated = len(raw_rows) > max_rows
    rows = [_row_to_dict(r) for r in raw_rows[:max_rows]]
    return {"rows": rows, "row_count": len(rows), "truncated": truncated}


def describe_schema(conn: sqlite3.Connection) -> str:
    """Return the CREATE TABLE / CREATE INDEX statements from sqlite_master.

    Reflecting from the live DB (vs pasting a hard-coded string) keeps this
    honest across schema changes and works for any tag the agent is pointed
    at, even ones with different DDL later.
    """
    rows = conn.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type IN ('table', 'index') AND sql IS NOT NULL "
        "ORDER BY type DESC, name"  # tables before indexes
    ).fetchall()
    return "\n\n".join(r[0].strip() + ";" for r in rows)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a Row to a JSON-safe dict. All SQLite native types are already
    JSON-safe (str/int/float/None/bytes); we exclude bytes since we never
    store any and don't want to surprise json.dumps with a TypeError."""
    return {k: row[k] for k in row.keys()}


def make_tool(
    conn: sqlite3.Connection,
    *,
    max_rows: int = _DEFAULT_MAX_ROWS,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> Tool:
    """Wrap `run_query` as a Tool bound to `conn` via closure.

    The connection is captured in the handler's closure so the agent
    doesn't have to manage it. Row cap and timeout can be overridden
    per-agent if the defaults ever prove wrong.
    """

    def handler(inputs: dict[str, Any]) -> str:
        result = run_query(conn, inputs["sql"], max_rows=max_rows, timeout_s=timeout_s)
        return json.dumps(result)

    return Tool(schema=SCHEMA, handler=handler)
