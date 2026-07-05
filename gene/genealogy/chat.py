"""Interactive chat REPL scoped to one loaded family.

    python -m gene.genealogy.chat bronte
    python -m gene.genealogy.chat bronte --ask "How many children did Patrick have?"
    python -m gene.genealogy.chat bronte --log

The tag must correspond to a SQLite DB built by `gene.genealogy.load`.
Passing an unknown tag surfaces a FileNotFoundError with the load
command in the message.

Mirrors `gene.agent.chat_loop` — the difference is the required `tag`
positional and the genealogy-specific factory that wires the run_query
tool + schema-aware system prompt.
"""

import argparse
from datetime import datetime
from pathlib import Path

from gene.genealogy.agent import build_conversation

_LOG_AUTO = "__auto__"


def chat(
    tag: str,
    model: str = "sonnet",
    verbose: bool = False,
    ask: str | None = None,
    log_path: Path | str | None = None,
) -> None:
    """Chat with the agent for the given family. `ask` runs one turn and exits."""
    conv = build_conversation(tag, model=model, log_path=log_path)
    log_note = f", log={log_path}" if log_path else ""
    print(f"Genealogy chat: tag={tag}, model={model}{log_note}. /quit to exit.\n")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="gene.genealogy.chat",
        description="Chat with the genealogy agent for a loaded family tree.",
    )
    parser.add_argument("tag", help="family tag (a DB built by gene.genealogy.load)")
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
            "logs/genealogy-{tag}-{YYYYMMDD-HHMMSS}.jsonl; --log PATH uses PATH."
        ),
    )
    args = parser.parse_args()

    log_path = args.log
    if log_path == _LOG_AUTO:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = f"logs/genealogy-{args.tag}-{ts}.jsonl"

    chat(
        tag=args.tag,
        model=args.model,
        verbose=args.verbose,
        ask=args.ask,
        log_path=log_path,
    )
