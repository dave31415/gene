"""GEDCOM → Pydantic models via ged4py.

The `ged4py` library is used here and nowhere else — every function in this
module returns models from `gene.genealogy.models`, so downstream code
never sees a ged4py type. That confinement is deliberate: it means the
choice of parser is swappable, and it keeps the SQLite loader and any
future consumers off the ged4py API.

Kept narrow — we extract only what the current schema stores. Names beyond
the first, notes, sources, addresses, and events beyond birth/death/burial/
marriage/divorce are dropped for now.
"""

import re
from pathlib import Path
from typing import Any

from ged4py.parser import GedcomReader

from gene.genealogy.models import Event, EventType, Family, Individual, Sex

_YEAR_RE = re.compile(r"\b(\d{3,4})\b")

_INDI_EVENTS = {"BIRT": EventType.BIRTH, "DEAT": EventType.DEATH, "BURI": EventType.BURIAL}
_FAM_EVENTS = {"MARR": EventType.MARRIAGE, "DIV": EventType.DIVORCE}


def parse_gedcom(path: str | Path) -> tuple[list[Individual], list[Family]]:
    """Parse a .ged file into lists of individuals and families."""
    path = Path(path)
    individuals: list[Individual] = []
    families: list[Family] = []
    with GedcomReader(str(path)) as reader:
        for rec in reader.records0("INDI"):
            individuals.append(_build_individual(rec))
        for rec in reader.records0("FAM"):
            families.append(_build_family(rec))
    return individuals, families


def _build_individual(rec: Any) -> Individual:
    given, surname, full = _parse_name(_first_sub(rec, "NAME"))
    events = [
        e
        for tag, etype in _INDI_EVENTS.items()
        if (e := _build_event(_first_sub(rec, tag), etype)) is not None
    ]
    return Individual(
        id=rec.xref_id,
        given=given,
        surname=surname,
        full_name=full,
        sex=_parse_sex(_first_value(rec, "SEX")),
        events=events,
    )


def _build_family(rec: Any) -> Family:
    events = [
        e
        for tag, etype in _FAM_EVENTS.items()
        if (e := _build_event(_first_sub(rec, tag), etype)) is not None
    ]
    return Family(
        id=rec.xref_id,
        husband_id=_first_value(rec, "HUSB"),
        wife_id=_first_value(rec, "WIFE"),
        children_ids=[s.value for s in rec.sub_records if s.tag == "CHIL" and s.value],
        events=events,
    )


def _first_sub(rec: Any, tag: str) -> Any:
    # Use raw sub_records — ged4py's sub_tags() dereferences xref pointers,
    # which loses the string id we need for links.
    for s in rec.sub_records:
        if s.tag == tag:
            return s
    return None


def _first_value(rec: Any, tag: str) -> str | None:
    sub = _first_sub(rec, tag)
    if sub is None or sub.value is None:
        return None
    return str(sub.value).strip() or None


def _parse_name(name_sub: Any) -> tuple[str | None, str | None, str | None]:
    """Return (given, surname, full). ged4py may give us a tuple or string."""
    if name_sub is None or name_sub.value is None:
        return None, None, None
    value = name_sub.value
    if isinstance(value, tuple) and len(value) >= 2:
        given = (value[0] or "").strip() or None
        surname = (value[1] or "").strip() or None
        full = " ".join(p for p in (given, f"/{surname}/" if surname else None) if p)
        return given, surname, full or None
    full = str(value).strip()
    m = re.match(r"^(.*?)\s*/([^/]*)/\s*(.*)$", full)
    if m:
        given = (m.group(1) + " " + m.group(3)).strip() or None
        surname = m.group(2).strip() or None
        return given, surname, full or None
    return None, None, full or None


def _parse_sex(value: str | None) -> Sex:
    if value == "M":
        return Sex.MALE
    if value == "F":
        return Sex.FEMALE
    return Sex.UNKNOWN


def _build_event(sub: Any, event_type: EventType) -> Event | None:
    if sub is None:
        return None
    date_raw = _first_value(sub, "DATE")
    place = _first_value(sub, "PLAC")
    if date_raw is None and place is None:
        return None
    year = None
    if date_raw:
        m = _YEAR_RE.search(date_raw)
        if m:
            year = int(m.group(1))
    return Event(type=event_type, date_raw=date_raw, date_year=year, place=place)
