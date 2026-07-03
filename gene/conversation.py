"""Stateful chat wrapper over `CachedAnthropic`.

`Conversation` holds the growing messages list for the life of the object.
No persistence — history disappears when the object is garbage-collected.
Each `ask()` sends the whole history, so the disk cache keys still work:
replay the same sequence of user inputs and every turn hits the cache.
"""

import argparse

from anthropic.types import Message

from gene.config import get_llm_config
from gene.llm import CachedAnthropic


class Conversation:
    """One chat session. Owns the messages list; delegates to a `CachedAnthropic`."""

    def __init__(self, llm: CachedAnthropic, system: str | None = None):
        self.llm = llm
        self.system = system
        self.history: list[dict] = []

    def ask(self, text: str) -> Message:
        """Add a user turn, send, remember the assistant turn, return the Message."""
        self.history.append({"role": "user", "content": text})
        msg = self.llm.send(messages=self.history, system=self.system)
        self.history.append({"role": "assistant", "content": msg.content})
        return msg


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
