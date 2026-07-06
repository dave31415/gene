"""System-prompt loading and rendering, use-case-agnostic.

Prompts live as `.md` files in a `prompts/` directory that each agent
package points at via a `PROMPTS_DIR` constant in its `__init__.py`:

    # gene/<domain>/__init__.py
    from pathlib import Path
    PROMPTS_DIR = Path(__file__).parent / "prompts"

Templates use `str.format` placeholders; the caller passes them as
keyword args to `render_prompt`.
"""

from pathlib import Path
from typing import Any


def load_prompt(name: str, prompts_dir: Path) -> str:
    """Read a system-prompt template by name from `prompts_dir`."""
    path = prompts_dir / f"{name}.md"
    if not path.exists():
        available = sorted(p.stem for p in prompts_dir.glob("*.md"))
        raise ValueError(
            f"Unknown prompt {name!r} in {prompts_dir}. Available: {available}"
        )
    return path.read_text()


def render_prompt(name: str, prompts_dir: Path, **placeholders: Any) -> str:
    """Load a named template and substitute keyword placeholders."""
    return load_prompt(name, prompts_dir).format(**placeholders)
