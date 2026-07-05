"""Tests for gene.genealogy.tools.query.

Builds a tiny SQLite DB in tmp_path and exercises the guards directly
against a real connection — no mocks. Timeout test uses a WITH RECURSIVE
that would run indefinitely without the interrupt.
"""

import json
import sqlite3

import pytest

from gene.agent.tool import Tool
from gene.genealogy.models import Event, EventType, Family, Individual, Sex
from gene.genealogy.store import build_db
from gene.genealogy.tools.query import describe_schema, make_tool, run_query


@pytest.fixture
def conn(tmp_path):
    individuals = [
        Individual(id="@I1@", given="David", surname="Johnston", full_name="David /Johnston/", sex=Sex.MALE,
                   events=[Event(type=EventType.BIRTH, date_raw="1975", date_year=1975, place="Quincy")]),
        Individual(id="@I2@", given="Mary", surname="Keenan", full_name="Mary /Keenan/", sex=Sex.FEMALE),
        Individual(id="@I3@", given="Alice", surname="Johnston", full_name="Alice /Johnston/", sex=Sex.FEMALE),
    ]
    families = [Family(id="@F1@", husband_id="@I1@", wife_id="@I2@", children_ids=["@I3@"])]
    db_path = tmp_path / "test.sqlite"
    build_db(individuals, families, db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


# ---------- happy path ----------


def test_run_query_returns_rows(conn):
    r = run_query(conn, "SELECT id, surname FROM individuals ORDER BY id")
    assert r["row_count"] == 3
    assert r["truncated"] is False
    assert r["rows"][0] == {"id": "@I1@", "surname": "Johnston"}


def test_run_query_accepts_with_clause(conn):
    r = run_query(conn, "WITH j AS (SELECT id FROM individuals WHERE surname = 'Johnston') SELECT COUNT(*) AS n FROM j")
    assert r["rows"] == [{"n": 2}]


def test_run_query_accepts_leading_whitespace_and_comments(conn):
    sql = "  -- pick a name\n  SELECT surname FROM individuals WHERE id = '@I1@'"
    r = run_query(conn, sql)
    assert r["rows"] == [{"surname": "Johnston"}]


# ---------- statement whitelist ----------


@pytest.mark.parametrize("sql", [
    "INSERT INTO individuals(id, sex) VALUES ('@X@', 'M')",
    "UPDATE individuals SET given = 'X' WHERE id = '@I1@'",
    "DELETE FROM individuals",
    "DROP TABLE individuals",
    "CREATE TABLE foo (a INT)",
])
def test_run_query_rejects_non_select(conn, sql):
    r = run_query(conn, sql)
    assert "error" in r
    assert "SELECT" in r["error"] or "WITH" in r["error"]


# ---------- keyword blacklist ----------


@pytest.mark.parametrize("sql", [
    "SELECT * FROM individuals; ATTACH DATABASE 'other.db' AS other",
    "SELECT load_extension('/tmp/evil.so')",
    "SELECT * FROM individuals WHERE id = '@I1@' PRAGMA foreign_keys",
])
def test_run_query_rejects_forbidden_keywords(conn, sql):
    r = run_query(conn, sql)
    assert "error" in r
    assert "not allowed" in r["error"]


# ---------- row cap ----------


def test_run_query_row_cap_and_truncated_flag(conn):
    r = run_query(conn, "SELECT id FROM individuals", max_rows=2)
    assert r["row_count"] == 2
    assert r["truncated"] is True


def test_run_query_at_cap_not_truncated(conn):
    r = run_query(conn, "SELECT id FROM individuals", max_rows=3)
    assert r["row_count"] == 3
    assert r["truncated"] is False


# ---------- error path ----------


def test_run_query_reports_sql_error(conn):
    r = run_query(conn, "SELECT nosuchcol FROM individuals")
    assert "error" in r
    assert "sql error" in r["error"]


# ---------- timeout ----------


def test_run_query_times_out(conn):
    # A recursive CTE with no base termination — would run until the row
    # cap without the interrupt. timeout_s=0.05 is well under fetchmany's
    # patience for 101 rows of a runaway generator.
    sql = "WITH RECURSIVE seq(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM seq) SELECT n FROM seq"
    r = run_query(conn, sql, max_rows=10_000_000, timeout_s=0.05)
    assert "error" in r
    assert "timed out" in r["error"]


# ---------- describe_schema ----------


def test_describe_schema_includes_all_tables(conn):
    text = describe_schema(conn)
    for name in ("individuals", "families", "family_children", "individual_events", "family_events"):
        assert f"CREATE TABLE {name}" in text


def test_describe_schema_includes_indexes(conn):
    text = describe_schema(conn)
    assert "CREATE INDEX" in text
    assert "idx_individuals_surname" in text


# ---------- make_tool ----------


def test_make_tool_returns_tool_with_schema(conn):
    tool = make_tool(conn)
    assert isinstance(tool, Tool)
    assert tool.schema["name"] == "run_query"
    assert "sql" in tool.schema["input_schema"]["properties"]
    assert tool.schema["input_schema"]["required"] == ["sql"]


def test_tool_handler_returns_rows_as_json(conn):
    tool = make_tool(conn)
    result = json.loads(tool.handler({"sql": "SELECT id FROM individuals ORDER BY id"}))
    assert result["row_count"] == 3
    assert result["rows"][0] == {"id": "@I1@"}
    assert result["truncated"] is False


def test_tool_handler_serializes_error_as_json(conn):
    tool = make_tool(conn)
    result = json.loads(tool.handler({"sql": "DROP TABLE individuals"}))
    assert "error" in result


def test_tool_binds_to_specific_connection(tmp_path):
    """Two tools on two DBs must not cross-contaminate."""
    build_db([Individual(id="@A@", sex=Sex.MALE)], [], tmp_path / "a.sqlite")
    build_db([Individual(id="@B@", sex=Sex.FEMALE)], [], tmp_path / "b.sqlite")
    conn_a = sqlite3.connect(f"file:{tmp_path / 'a.sqlite'}?mode=ro", uri=True)
    conn_a.row_factory = sqlite3.Row
    conn_b = sqlite3.connect(f"file:{tmp_path / 'b.sqlite'}?mode=ro", uri=True)
    conn_b.row_factory = sqlite3.Row

    a_rows = json.loads(make_tool(conn_a).handler({"sql": "SELECT id FROM individuals"}))["rows"]
    b_rows = json.loads(make_tool(conn_b).handler({"sql": "SELECT id FROM individuals"}))["rows"]

    assert a_rows == [{"id": "@A@"}]
    assert b_rows == [{"id": "@B@"}]
