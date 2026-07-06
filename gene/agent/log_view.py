"""Human-readable view of a JSONL turn log.

The log itself is the source of truth — raw request/response payloads,
timings, tool I/O. This module renders content-focused (default) and
step-by-step (`--trace`) views over a whole file.

    uv run python -m gene.agent.log_view                    # newest file in logs/
    uv run python -m gene.agent.log_view PATH               # index / in / out per turn
    uv run python -m gene.agent.log_view PATH --trace       # full trace per turn
    uv run python -m gene.agent.log_view PATH --turn N      # one turn (negatives ok)
    uv run python -m gene.agent.log_view PATH --tail K      # last K turns
"""

import argparse
import json
import sys
from pathlib import Path

from gene.agent.ansi import paint
from gene.agent.turn import Turn

_RULE_WIDTH = 60
_MAX_CONTENT = 200
_LOGS_DIR = Path("logs")


def _header(index: int) -> str:
    return paint(f" Turn number: {index} ".center(_RULE_WIDTH, "="), "bold", "cyan")


def latest_log(logs_dir: Path = _LOGS_DIR) -> Path | None:
    """Newest `*.jsonl` in `logs_dir` by mtime, or None if none exist."""
    if not logs_dir.is_dir():
        return None
    candidates = list(logs_dir.glob("*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_turns(path: Path) -> list[Turn]:
    """Read a JSONL file into Turns. Malformed lines are skipped with a warning."""
    turns: list[Turn] = []
    for i, raw in enumerate(path.read_text().splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            turns.append(Turn.from_dict(json.loads(raw)))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"warning: skipping line {i}: {type(e).__name__}: {e}", file=sys.stderr)
    return turns


def _truncate(s: str, n: int = _MAX_CONTENT) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def _body_default(turn: Turn) -> str:
    """Content-focused body: user input + assistant reply, truncated."""
    out = turn.text or f"(no reply — reason={turn.terminal_reason})"
    in_label = paint("In: ", "cyan")
    out_label = paint("Out:", "cyan")
    return f"  {in_label} {_truncate(turn.user_input)}\n  {out_label} {_truncate(out)}"


def render(indexed: list[tuple[int, Turn]], trace: bool) -> str:
    """Content-focused body by default; per-step trace with --trace.

    Every block is prefixed by a rule header carrying the original
    file-position index, so it stays visible under `--turn N` / `--tail K`.
    """
    blocks = []
    for i, t in indexed:
        body = t.trace() if trace else _body_default(t)
        blocks.append(f"{_header(i)}\n{body}")
    return "\n\n".join(blocks)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="View a gene turn log (JSONL).")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="Path to the JSONL log file. Defaults to the newest file in logs/.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print full per-step trace for each turn (default: index / in / out).",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--turn",
        type=int,
        metavar="N",
        help="Only render the turn at index N. Negatives count from the end.",
    )
    selection.add_argument(
        "--tail",
        type=int,
        metavar="K",
        help="Only render the last K turns.",
    )
    args = parser.parse_args(argv)

    path = args.path
    if path is None:
        path = latest_log()
        if path is None:
            print(f"error: no *.jsonl files in {_LOGS_DIR}/", file=sys.stderr)
            sys.exit(1)
        print(paint(f"# using {path}", "dim"), file=sys.stderr)

    turns = load_turns(path)
    if not turns:
        return

    indexed = list(enumerate(turns))

    if args.turn is not None:
        try:
            indexed = [indexed[args.turn]]
        except IndexError:
            print(
                f"error: --turn {args.turn} out of range (0..{len(turns) - 1})",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.tail is not None:
        indexed = indexed[-args.tail :]

    print(render(indexed, trace=args.trace))


if __name__ == "__main__":
    main()
