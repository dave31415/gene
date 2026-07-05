"""Tiny ANSI color helpers.

`paint(text, *styles)` wraps text in ANSI codes when colors are enabled,
returning the raw text otherwise. Enabled iff stdout is a TTY and the
`NO_COLOR` env var is unset. Detection happens on import — this is a CLI
helper, so that's fine.
"""

import os
import sys

_ENABLED = sys.stdout.isatty() and "NO_COLOR" not in os.environ

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
}


def paint(text: str, *styles: str) -> str:
    if not _ENABLED or not styles:
        return text
    prefix = "".join(_CODES[s] for s in styles)
    return f"{prefix}{text}{_CODES['reset']}"
