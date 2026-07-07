"""Interactive chat REPL for gene.

Wires a `TurnRunner` (with the calculator tool by default) into a
`Conversation` and pumps stdin through it. Pass `--verbose` to print
`turn.summary()` after each response — handy for seeing when tools ran.
The default system prompt lives at `prompts/general_purpose.md`; pass
`--system STRING` to override.
"""

import argparse
from datetime import datetime
from pathlib import Path

from gene.agent import PROMPTS_DIR
from gene.agent.config import get_llm_config
from gene.agent.conversation import Conversation
from gene.agent.llm import CachedAnthropic
from gene.agent.prompts import render_prompt
from gene.agent.tools.calculator import CALCULATOR
from gene.agent.turn_runner import TurnRunner

_LOG_AUTO = "__auto__"  # argparse sentinel: --log passed without a value


def chat_loop(
    model: str = "sonnet",
    system: str | None = None,
    verbose: bool = False,
    log_path: Path | str | None = None,
) -> None:

    """Interactive REPL. /quit or Ctrl-D to exit."""
    if system is None:
        system = render_prompt("general_purpose", PROMPTS_DIR)

    llm = CachedAnthropic(config=get_llm_config(model=model))
    runner = TurnRunner(llm=llm, tools=[CALCULATOR])
    conv = Conversation(runner, system=system, log_path=log_path)

    log_note = f", log={log_path}" if log_path else ""
    print(f"Chat started (model={model}, tools=[calculator]{log_note}). /quit to exit.\n")

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
    parser = argparse.ArgumentParser(description="Chat REPL for gene.")
    parser.add_argument(
        "--model",
        choices=["haiku", "sonnet", "opus"],
        default="sonnet",
        help="Model tag (default: sonnet)",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Override the default system prompt (prompts/general_purpose.md).",
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
            "logs/chat-{YYYYMMDD-HHMMSS}.jsonl; --log PATH uses PATH."
        ),
    )
    args = parser.parse_args()

    log_path = args.log
    if log_path == _LOG_AUTO:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = f"logs/chat-{ts}.jsonl"

    chat_loop(model=args.model, system=args.system, verbose=args.verbose, log_path=log_path)
