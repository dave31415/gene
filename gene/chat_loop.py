"""Interactive chat REPL for gene.

Wires a `TurnRunner` (with the calculator tool by default) into a
`Conversation` and pumps stdin through it. Pass `--verbose` to print
`turn.summary()` after each response — handy for seeing when tools ran.
"""

import argparse

from gene.config import get_llm_config
from gene.conversation import Conversation
from gene.llm import CachedAnthropic
from gene.tools.calculator import CALCULATOR
from gene.turn_runner import TurnRunner


def chat_loop(
    model: str = "sonnet",
    system: str | None = None,
    verbose: bool = False,
) -> None:
    """Interactive REPL. /quit or Ctrl-D to exit."""
    llm = CachedAnthropic(config=get_llm_config(model=model))
    runner = TurnRunner(llm=llm, tools=[CALCULATOR])
    conv = Conversation(runner, system=system)
    print(f"Chat started (model={model}, tools=[calculator]). /quit to exit.\n")

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
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print turn summary (steps, tools, tokens) after each response.",
    )
    args = parser.parse_args()
    chat_loop(model=args.model, system=args.system, verbose=args.verbose)
