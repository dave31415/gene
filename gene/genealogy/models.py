"""Domain models for parsed GEDCOM records.

Thin — one Pydantic model per SQLite table, no more. `ged4py` types stay
inside `gedcom.py`; everything crossing the parser boundary is one of these.
Deliberately narrow on day one: skips notes, sources, ADDR/PLAC split,
approximate-date flags, and secondary events. Add fields when a real query
needs them.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class Sex(StrEnum):
    MALE = "M"
    FEMALE = "F"
    UNKNOWN = "U"


class EventType(StrEnum):
    BIRTH = "BIRT"
    DEATH = "DEAT"
    BURIAL = "BURI"
    MARRIAGE = "MARR"
    DIVORCE = "DIV"


class Event(BaseModel):
    """A dated, placed occurrence attached to an individual or family.

    `date_raw` preserves the source string; `date_year` is the extracted int
    used for range queries. Both may be None (event known to have happened
    but undated). Owner (individual/family) is tracked by the collection
    that holds the event, not on the event itself.
    """

    type: EventType
    date_raw: str | None = None
    date_year: int | None = None
    place: str | None = None


class Individual(BaseModel):
    """One INDI record. Events list holds birth/death/burial only for now."""

    id: str
    given: str | None = None
    surname: str | None = None
    full_name: str | None = None
    sex: Sex = Sex.UNKNOWN
    events: list[Event] = Field(default_factory=list)


class Family(BaseModel):
    """One FAM record. Spouse links are direct; children as a list of xrefs.

    Events list holds marriage/divorce only for now.
    """

    id: str
    husband_id: str | None = None
    wife_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
