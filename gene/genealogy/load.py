"""CLI: parse a GEDCOM file by family_tag and write the corresponding SQLite DB.

    python -m gene.genealogy.load bronte

Reads `genealogy_data/<family_tag>.ged`, writes
`gene/genealogy/db/<family_tag>.sqlite` (rebuilding it if it already exists).
"""

import argparse
import sys

from gene.genealogy.config import (
    available_family_tags,
    get_db_path,
    get_gedcom_file,
)
from gene.genealogy.gedcom import parse_gedcom
from gene.genealogy.store import build_db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gene.genealogy.load",
        description="Parse a GEDCOM file and load it into SQLite.",
    )
    parser.add_argument(
        "family_tag",
        help=f"gedcom family_tag. available: {available_family_tags()}",
    )
    args = parser.parse_args(argv)

    try:
        ged_path = get_gedcom_file(args.family_tag)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    db_path = get_db_path(args.family_tag)
    print(f"parsing {ged_path}")
    individuals, families = parse_gedcom(ged_path)
    print(f"  {len(individuals)} individuals, {len(families)} families")
    print(f"writing {db_path}")
    build_db(individuals, families, db_path)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
