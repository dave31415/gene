"""Stateful chat wrapper over `CachedAnthropic`.

`Conversation` holds the growing messages list for the life of the object.
No persistence — history disappears when the object is garbage-collected.
Each `ask()` sends the whole history, so the disk cache keys still work:
replay the same sequence of user inputs and every turn hits the cache.
"""

from anthropic.types import Message
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
