"""SQLite storage for parsed GEDCOM data.

5-table 3NF schema — see `_SCHEMA` below. `build_db` drops any existing
file and rebuilds from scratch; there is no incremental update path.

`open_db` returns a read-only connection for query use. Writers should go
through `build_db` only, so the database on disk is always a snapshot of
one parse pass.
"""

import sqlite3
from pathlib import Path
from typing import Iterable

from gene.genealogy.config import get_db_path
from gene.genealogy.models import Family, Individual

_SCHEMA = """
CREATE TABLE individuals (
    id         TEXT PRIMARY KEY,
    given      TEXT,
    surname    TEXT,
    full_name  TEXT,
    sex        TEXT NOT NULL CHECK (sex IN ('M', 'F', 'U'))
);

CREATE TABLE families (
    id          TEXT PRIMARY KEY,
    husband_id  TEXT REFERENCES individuals(id),
    wife_id     TEXT REFERENCES individuals(id)
);

CREATE TABLE family_children (
    family_id      TEXT NOT NULL REFERENCES families(id),
    individual_id  TEXT NOT NULL REFERENCES individuals(id),
    PRIMARY KEY (family_id, individual_id)
);

CREATE TABLE individual_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    individual_id  TEXT NOT NULL REFERENCES individuals(id),
    type           TEXT NOT NULL,
    date_raw       TEXT,
    date_year      INTEGER,
    place          TEXT
);

CREATE TABLE family_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id  TEXT NOT NULL REFERENCES families(id),
    type       TEXT NOT NULL,
    date_raw   TEXT,
    date_year  INTEGER,
    place      TEXT
);

CREATE INDEX idx_individuals_surname ON individuals(surname);
CREATE INDEX idx_individuals_given ON individuals(given);
CREATE INDEX idx_individual_events_year ON individual_events(date_year);
CREATE INDEX idx_individual_events_individual ON individual_events(individual_id);
CREATE INDEX idx_family_events_family ON family_events(family_id);
CREATE INDEX idx_family_children_individual ON family_children(individual_id);
"""


def build_db(
    individuals: Iterable[Individual],
    families: Iterable[Family],
    db_path: Path,
) -> None:
    """Drop any existing DB at `db_path` and rebuild it from the given records.

    Families are inserted before children/spouse links resolve, so
    referential integrity is enforced at end-of-transaction only (deferred
    would require SQLite's deferred FK feature; instead we just insert in
    an order that satisfies the constraints: individuals → families →
    children join → events).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA)
        _insert_individuals(conn, individuals)
        _insert_families(conn, families)
        conn.commit()


def open_db(tag: str) -> sqlite3.Connection:
    """Open the database for `tag` in read-only mode."""
    path = get_db_path(tag)
    if not path.exists():
        raise FileNotFoundError(
            f"no database for tag {tag!r} at {path}. "
            f"run: python -m gene.genealogy.load {tag}"
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _insert_individuals(conn: sqlite3.Connection, individuals: Iterable[Individual]) -> None:
    indi_rows = []
    event_rows = []
    for i in individuals:
        indi_rows.append((i.id, i.given, i.surname, i.full_name, i.sex.value))
        for e in i.events:
            event_rows.append((i.id, e.type.value, e.date_raw, e.date_year, e.place))
    conn.executemany(
        "INSERT INTO individuals(id, given, surname, full_name, sex) VALUES (?, ?, ?, ?, ?)",
        indi_rows,
    )
    conn.executemany(
        "INSERT INTO individual_events(individual_id, type, date_raw, date_year, place) "
        "VALUES (?, ?, ?, ?, ?)",
        event_rows,
    )


def _insert_families(conn: sqlite3.Connection, families: Iterable[Family]) -> None:
    fam_rows = []
    child_rows = []
    event_rows = []
    for f in families:
        fam_rows.append((f.id, f.husband_id, f.wife_id))
        for cid in f.children_ids:
            child_rows.append((f.id, cid))
        for e in f.events:
            event_rows.append((f.id, e.type.value, e.date_raw, e.date_year, e.place))
    conn.executemany(
        "INSERT INTO families(id, husband_id, wife_id) VALUES (?, ?, ?)",
        fam_rows,
    )
    conn.executemany(
        "INSERT INTO family_children(family_id, individual_id) VALUES (?, ?)",
        child_rows,
    )
    conn.executemany(
        "INSERT INTO family_events(family_id, type, date_raw, date_year, place) "
        "VALUES (?, ?, ?, ?, ?)",
        event_rows,
    )
