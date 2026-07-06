"""Paths for genealogy data and databases.

GEDCOM sources live in `genealogy_data/` at the repo root (gitignored — they
may be private). Each file is addressed by its stem, e.g. `bronte.ged` is
family_tag `bronte`. Built SQLite databases live in `gene/genealogy/db/`
(also gitignored) as `<family_tag>.sqlite`.

Kept as functions so callers can override via env / CLI later without
touching module state.
"""

from pathlib import Path


def get_gedcom_data_dir() -> Path:
    """Directory holding the source .ged files."""
    return Path(__file__).resolve().parent.parent.parent / "genealogy_data"


def available_family_tags() -> list[str]:
    """Sorted list of family_tags for every .ged in the data dir."""
    return sorted(p.stem for p in get_gedcom_data_dir().glob("*.ged"))


def get_gedcom_file(family_tag: str) -> Path:
    """Path to the .ged for `family_tag`. Raises ValueError if no such file exists."""
    path = get_gedcom_data_dir() / f"{family_tag}.ged"
    if not path.exists():
        family_tags = available_family_tags()
        raise ValueError(
            f"unknown gedcom family_tag {family_tag!r}. available: {family_tags}"
        )
    return path


def get_db_dir() -> Path:
    """Directory for built SQLite databases. Created on demand."""
    d = Path(__file__).resolve().parent / "db"
    d.mkdir(exist_ok=True)
    return d


def get_db_path(family_tag: str) -> Path:
    """Path to the SQLite file for `family_tag`. Does not check existence — a
    caller building the DB expects it not to exist yet."""
    return get_db_dir() / f"{family_tag}.sqlite"
