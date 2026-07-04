"""Tool types shared by the runner and individual tool modules.

Kept in its own file so tool modules (`gene/tools/*.py`) never import
from `turn.py` or `conversation.py` — clean layering with no cycles.

`Tool` bundles the Anthropic tool schema with a handler. `ToolCall` is
the observability record: what the model asked for, what came back,
and how long it took.
"""

from collections.abc import Callable
from typing import Any, NamedTuple


class Tool(NamedTuple):
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]


class ToolCall(NamedTuple):
    tool_use_id: str
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool
    seconds: float
