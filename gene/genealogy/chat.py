"""Interactive chat REPL scoped to one loaded family.

    python -m gene.genealogy.chat bronte
    python -m gene.genealogy.chat bronte --ask "How many children did Patrick have?"
    python -m gene.genealogy.chat bronte --log

The family_tag must correspond to a SQLite DB built by `gene.genealogy.load`.
Unknown family_tags and not-yet-loaded family_tags both exit 2 with a clean
stderr message (no traceback) — see `_run_cli`.

Mirrors `gene.agent.chat_loop` — the difference is the required
`family_tag` positional and the genealogy-specific factory that wires the
run_query tool + schema-aware system prompt.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from gene.genealogy.agent import build_conversation
from gene.genealogy.config import available_family_tags

_LOG_AUTO = "__auto__"


def chat(
    family_tag: str,
    model: str = "sonnet",
    verbose: bool = False,
    ask: str | None = None,
    log_path: Path | str | None = None,
) -> None:
    """Chat with the agent for the given family. `ask` runs one turn and exits."""
    conv = build_conversation(family_tag, model=model, log_path=log_path)
    log_note = f", log={log_path}" if log_path else ""
    print(
        f"Genealogy chat: family_tag={family_tag}, model={model}{log_note}."
        " /quit to exit.\n"
    )

    if ask is not None:
        turn = conv.ask(ask)
        print(f"llm> {turn.text}")
        if verbose:
            print(f"     [{turn.summary()}]")
        return

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            break
        turn = conv.ask(user)
        print(f"llm> {turn.text}\n")
        if verbose:
            print(f"     [{turn.summary()}]\n")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gene.genealogy.chat",
        description="Chat with the genealogy agent for a loaded family tree.",
    )
    parser.add_argument(
        "family_tag", help="family tag (a DB built by gene.genealogy.load)"
    )
    parser.add_argument(
        "--model",
        choices=["haiku", "sonnet", "opus"],
        default="sonnet",
        help="Model tag (default: sonnet)",
    )
    parser.add_argument(
        "--ask",
        default=None,
        metavar="QUESTION",
        help="One-shot: run a single question and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print turn summary (steps, tools, tokens) after each response.",
    )
    parser.add_argument(
        "--log",
        nargs="?",
        const=_LOG_AUTO,
        default=None,
        metavar="PATH",
        help=(
            "Log turns as JSONL. Bare --log auto-generates "
            "logs/genealogy-{family_tag}-{YYYYMMDD-HHMMSS}.jsonl; "
            "--log PATH uses PATH."
        ),
    )
    return parser


def _resolve_log_path(log_arg: str | None, family_tag: str) -> str | None:
    if log_arg != _LOG_AUTO:
        return log_arg
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"logs/genealogy-{family_tag}-{ts}.jsonl"


def _run_cli(args: argparse.Namespace) -> None:
    """Invoke `chat` and translate a missing-DB error into a friendly exit."""
    try:
        chat(
            family_tag=args.family_tag,
            model=args.model,
            verbose=args.verbose,
            ask=args.ask,
            log_path=_resolve_log_path(args.log, args.family_tag),
        )
    except FileNotFoundError:
        # Distinguish "not a family at all" from "family exists but hasn't
        # been loaded yet" so the recovery hint is actionable.
        family_tags = available_family_tags()
        if args.family_tag not in family_tags:
            print(
                f"error: unknown family_tag {args.family_tag!r}."
                f" available: {family_tags}",
                file=sys.stderr,
            )
        else:
            print(
                f"error: family {args.family_tag!r} not built yet. "
                f"run: python -m gene.genealogy.load {args.family_tag}",
                file=sys.stderr,
            )
        sys.exit(2)


if __name__ == "__main__":
    _run_cli(_make_parser().parse_args())
