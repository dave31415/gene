"""Arithmetic sanity cases: basic math with unambiguous single-number answers."""

from gene.agent.eval_case import Case, text

CASES: list[Case] = [
    Case(
        name="addition",
        prompt="What is 137 + 264? Answer with just the number.",
        check=lambda m: "401" in text(m),
    ),
    Case(
        name="multiplication",
        prompt="What is 12 * 13? Answer with just the number.",
        check=lambda m: "156" in text(m),
    ),
    Case(
        name="subtraction",
        prompt="What is 1000 - 347? Answer with just the number.",
        check=lambda m: "653" in text(m),
    ),
]
