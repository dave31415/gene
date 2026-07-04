"""
A chat loop
"""

import argparse
from anthropic.types import Message

from gene.config import get_llm_config
from gene.conversation import Conversation
from gene.llm import CachedAnthropic


def _text(msg: Message) -> str:
    """Join every text block in a response into one string."""
    return "".join(b.text for b in msg.content if b.type == "text")


def chat_loop(model: str = "sonnet", system: str | None = None) -> None:
    """Interactive REPL. /quit or Ctrl-D to exit."""
    llm = CachedAnthropic(config=get_llm_config(model=model))
    conv = Conversation(llm, system=system)
    print(f"Chat started (model={model}). /quit to exit.\n")

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
        msg = conv.ask(user)
        print(f"llm> {_text(msg)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat REPL for gene.")
    parser.add_argument(
        "--model",
        choices=["haiku", "sonnet", "opus"],
        default="sonnet",
        help="Model tag (default: sonnet)",
    )
    parser.add_argument("--system", default=None, help="Optional system prompt")
    args = parser.parse_args()
    chat_loop(model=args.model, system=args.system)
