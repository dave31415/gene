"""Basic sanity cases: short factual questions with unambiguous answers."""

from gene.eval_case import Case, text

CASES: list[Case] = [
    Case(
        name="brian_boru_death",
        prompt="What year did Brian Boru die? Answer with just the year.",
        check=lambda m: "1014" in text(m),
    ),
    Case(
        name="magna_carta",
        prompt="What year was Magna Carta signed?",
        check=lambda m: "1215" in text(m),
    ),
    Case(
        name="simple_math",
        prompt="What is 7 * 8? Answer with just the number.",
        check=lambda m: "56" in text(m),
    ),
]
