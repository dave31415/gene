"""Small text-only predicates for eval-case `check` functions.

These operate on plain strings (typically `Turn.text` or the text of a
`Message` via `text()`), so they can be used by any case type without
pulling in `Turn`. Turn-inspecting predicates live in
`eval_predicates_turn`.

    check=lambda t: contains_all(t.text, ["John Dennis", "Mary Catherine"])
"""


def contains_all(text: str, needles: list[str]) -> bool:
    """True iff every needle appears in text (case-sensitive substring)."""
    return all(n in text for n in needles)


def contains_any(text: str, needles: list[str]) -> bool:
    return any(n in text for n in needles)
