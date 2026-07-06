"""Tests for gene.genealogy.agent.

Focuses on the system-prompt builder — the only unit-testable piece of
the factory without an LLM. `build_conversation` end-to-end needs a live
API connection, so it's exercised by the CLI, not here. Tests for the
`run_query` tool itself live in tests/test_query.py alongside the tool.
"""

import sqlite3

import pytest

from gene.genealogy.agent import build_system_prompt
from gene.genealogy.models import Family, Individual, Sex
from gene.genealogy.store import build_db


@pytest.fixture
def conn(tmp_path):
    individuals = [
        Individual(id="@I1@", given="David", surname="Johnston", sex=Sex.MALE),
        Individual(id="@I2@", given="Mary", surname="Keenan", sex=Sex.FEMALE),
    ]
    families = [Family(id="@F1@", husband_id="@I1@", wife_id="@I2@")]
    build_db(individuals, families, tmp_path / "t.sqlite")
    c = sqlite3.connect(f"file:{tmp_path / 't.sqlite'}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def test_system_prompt_includes_family_tag(conn):
    prompt = build_system_prompt("bronte", conn)
    assert "bronte" in prompt


def test_system_prompt_embeds_reflected_schema(conn):
    prompt = build_system_prompt("bronte", conn)
    for tbl in ("individuals", "families", "family_children", "individual_events", "family_events"):
        assert f"CREATE TABLE {tbl}" in prompt


def test_system_prompt_includes_domain_conventions(conn):
    """A few substrings that should be there — these are the load-bearing
    hints the agent needs to write correct SQL. If someone edits the
    template and drops one of these by accident, this test catches it."""
    prompt = build_system_prompt("bronte", conn)
    assert "@I1@" in prompt  # xref format shown
    assert "date_year" in prompt  # column semantics
    assert "run_query" in prompt  # tool named
