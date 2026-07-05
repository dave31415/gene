"""Tests for the ged4py wrapper.

Uses an inline GEDCOM snippet written to `tmp_path` so tests don't
depend on any real file in `genealogy_data/`.
"""

import pytest

from gene.genealogy.gedcom import parse_gedcom
from gene.genealogy.models import EventType, Sex

_SAMPLE = """\
0 HEAD
1 SOUR TestExporter
1 GEDC
2 VERS 5.5
2 FORM LINEAGE-LINKED
1 CHAR UTF-8
0 @I1@ INDI
1 NAME David /Johnston/
1 SEX M
1 BIRT
2 DATE 10 JAN 1975
2 PLAC Quincy
1 FAMC @F1@
0 @I2@ INDI
1 NAME Mary /Keenan/
1 SEX F
1 DEAT
2 DATE 1 FEB 2010
1 FAMS @F1@
0 @I3@ INDI
1 NAME Alice /Johnston/
1 SEX F
1 FAMC @F1@
0 @F1@ FAM
1 HUSB @I1@
1 WIFE @I2@
1 CHIL @I3@
1 MARR
2 DATE 1 MAY 2000
2 PLAC Boston
0 TRLR
"""


@pytest.fixture
def ged_path(tmp_path):
    p = tmp_path / "sample.ged"
    p.write_text(_SAMPLE, encoding="utf-8")
    return p


def test_parse_returns_expected_counts(ged_path):
    individuals, families = parse_gedcom(ged_path)
    assert len(individuals) == 3
    assert len(families) == 1


def test_parse_individual_fields(ged_path):
    individuals, _ = parse_gedcom(ged_path)
    by_id = {i.id: i for i in individuals}
    david = by_id["@I1@"]
    assert david.given == "David"
    assert david.surname == "Johnston"
    assert david.sex is Sex.MALE


def test_parse_extracts_birth_event(ged_path):
    individuals, _ = parse_gedcom(ged_path)
    david = next(i for i in individuals if i.id == "@I1@")
    birth = next(e for e in david.events if e.type is EventType.BIRTH)
    assert birth.date_year == 1975
    assert birth.place == "Quincy"
    assert "1975" in (birth.date_raw or "")


def test_parse_extracts_death_event(ged_path):
    individuals, _ = parse_gedcom(ged_path)
    mary = next(i for i in individuals if i.id == "@I2@")
    death = next(e for e in mary.events if e.type is EventType.DEATH)
    assert death.date_year == 2010
    assert death.place is None


def test_parse_skips_events_without_date_or_place(ged_path):
    individuals, _ = parse_gedcom(ged_path)
    alice = next(i for i in individuals if i.id == "@I3@")
    assert alice.events == []


def test_parse_family_links(ged_path):
    _, families = parse_gedcom(ged_path)
    fam = families[0]
    assert fam.id == "@F1@"
    assert fam.husband_id == "@I1@"
    assert fam.wife_id == "@I2@"
    assert fam.children_ids == ["@I3@"]


def test_parse_family_marriage_event(ged_path):
    _, families = parse_gedcom(ged_path)
    fam = families[0]
    marr = next(e for e in fam.events if e.type is EventType.MARRIAGE)
    assert marr.date_year == 2000
    assert marr.place == "Boston"


def test_sex_defaults_to_unknown_when_missing(tmp_path):
    p = tmp_path / "nosex.ged"
    p.write_text(
        "0 HEAD\n1 CHAR UTF-8\n0 @I1@ INDI\n1 NAME X /Y/\n0 TRLR\n",
        encoding="utf-8",
    )
    individuals, _ = parse_gedcom(p)
    assert individuals[0].sex is Sex.UNKNOWN
