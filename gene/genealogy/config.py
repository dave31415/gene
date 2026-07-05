"""Paths for genealogy data and databases.

GEDCOM sources live in `genealogy_data/` at the repo root (gitignored — they
may be private). Each file is addressed by its stem, e.g. `bronte.ged` is
tag `bronte`. Built SQLite databases live in `gene/genealogy/db/` (also
gitignored) as `<tag>.sqlite`.

Kept as functions so callers can override via env / CLI later without
touching module state.
"""

from pathlib import Path


def get_gedcom_data_dir() -> Path:
    """Directory holding the source .ged files."""
    return Path(__file__).resolve().parent.parent.parent / "genealogy_data"


def available_tags() -> list[str]:
    """Sorted list of tags for every .ged in the data dir."""
    return sorted(p.stem for p in get_gedcom_data_dir().glob("*.ged"))


def get_gedcom_file(tag: str) -> Path:
    """Path to the .ged for `tag`. Raises ValueError if no such file exists."""
    path = get_gedcom_data_dir() / f"{tag}.ged"
    if not path.exists():
        tags = available_tags()
        raise ValueError(f"unknown gedcom tag {tag!r}. available: {tags}")
    return path


def get_db_dir() -> Path:
    """Directory for built SQLite databases. Created on demand."""
    d = Path(__file__).resolve().parent / "db"
    d.mkdir(exist_ok=True)
    return d


def get_db_path(tag: str) -> Path:
    """Path to the SQLite file for `tag`. Does not check existence — a
    caller building the DB expects it not to exist yet."""
    return get_db_dir() / f"{tag}.sqlite"
