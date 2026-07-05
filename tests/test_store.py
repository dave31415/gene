"""Tests for the SQLite store.

Build the database from a hand-constructed list of models rather than
parsing a real .ged — keeps the store tests independent of the parser
so a parser regression can't cascade here.
"""

import sqlite3

import pytest

from gene.genealogy.models import Event, EventType, Family, Individual, Sex
from gene.genealogy.store import build_db


@pytest.fixture
def sample_records():
    individuals = [
        Individual(
            id="@I1@",
            given="David",
            surname="Johnston",
            full_name="David /Johnston/",
            sex=Sex.MALE,
            events=[
                Event(type=EventType.BIRTH, date_raw="10 JAN 1975", date_year=1975, place="Quincy"),
            ],
        ),
        Individual(
            id="@I2@",
            given="Mary",
            surname="Keenan",
            full_name="Mary /Keenan/",
            sex=Sex.FEMALE,
        ),
        Individual(id="@I3@", given="Alice", surname="Johnston", full_name="Alice /Johnston/", sex=Sex.FEMALE),
    ]
    families = [
        Family(
            id="@F1@",
            husband_id="@I1@",
            wife_id="@I2@",
            children_ids=["@I3@"],
            events=[
                Event(type=EventType.MARRIAGE, date_raw="1 MAY 2000", date_year=2000, place="Boston"),
            ],
        )
    ]
    return individuals, families


@pytest.fixture
def db_path(tmp_path, sample_records):
    path = tmp_path / "test.sqlite"
    individuals, families = sample_records
    build_db(individuals, families, path)
    return path


def _rows(conn, sql, *args):
    return conn.execute(sql, args).fetchall()


def test_build_db_creates_file(db_path):
    assert db_path.exists()


def test_individuals_row_count(db_path):
    with sqlite3.connect(db_path) as conn:
        (n,) = _rows(conn, "SELECT COUNT(*) FROM individuals")[0]
    assert n == 3


def test_individual_fields_roundtrip(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = _rows(conn, "SELECT * FROM individuals WHERE id = ?", "@I1@")[0]
    assert row["given"] == "David"
    assert row["surname"] == "Johnston"
    assert row["full_name"] == "David /Johnston/"
    assert row["sex"] == "M"


def test_family_and_children_links(db_path):
    with sqlite3.connect(db_path) as conn:
        fam = _rows(conn, "SELECT husband_id, wife_id FROM families WHERE id = ?", "@F1@")[0]
        children = _rows(
            conn,
            "SELECT individual_id FROM family_children WHERE family_id = ?",
            "@F1@",
        )
    assert fam == ("@I1@", "@I2@")
    assert [c[0] for c in children] == ["@I3@"]


def test_individual_events_inserted(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = _rows(
            conn,
            "SELECT type, date_year, place FROM individual_events WHERE individual_id = ?",
            "@I1@",
        )
    assert len(rows) == 1
    assert rows[0]["type"] == "BIRT"
    assert rows[0]["date_year"] == 1975
    assert rows[0]["place"] == "Quincy"


def test_family_events_inserted(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = _rows(
            conn,
            "SELECT type, date_year, place FROM family_events WHERE family_id = ?",
            "@F1@",
        )
    assert len(rows) == 1
    assert rows[0]["type"] == "MARR"
    assert rows[0]["date_year"] == 2000


def test_build_db_is_idempotent(tmp_path, sample_records):
    """Rebuilding over an existing DB should replace, not append."""
    individuals, families = sample_records
    path = tmp_path / "rebuild.sqlite"
    build_db(individuals, families, path)
    build_db(individuals, families, path)
    with sqlite3.connect(path) as conn:
        (n,) = _rows(conn, "SELECT COUNT(*) FROM individuals")[0]
    assert n == 3


def test_foreign_key_constraint_rejects_dangling_child(tmp_path):
    """A child xref pointing at no INDI record should be caught by the FK."""
    individuals = [Individual(id="@I1@", sex=Sex.MALE)]
    families = [Family(id="@F1@", children_ids=["@GHOST@"])]
    with pytest.raises(sqlite3.IntegrityError):
        build_db(individuals, families, tmp_path / "bad.sqlite")
