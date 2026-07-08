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


# Approximate Anthropic list prices in $/Mtok (input, output). Matched by
# longest-prefix on the model id from the stored request; unknown models
# fall through to (0, 0) and the report shows "?".
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4": (1.0, 5.0),
}


def _price_for(model: str) -> tuple[float, float]:
    for key in sorted(_PRICING, key=len, reverse=True):
        if model.startswith(key):
            return _PRICING[key]
    return (0.0, 0.0)


def _turn_cost(turn: Turn) -> tuple[float, float, str]:
    """Return (input_cost, output_cost, model_id) using actual token counts."""
    if not turn.steps:
        return 0.0, 0.0, "?"
    model = turn.steps[0].request.get("model", "?")
    in_rate, out_rate = _price_for(model)
    in_tok = sum(s.input_tokens for s in turn.steps)
    out_tok = sum(s.output_tokens for s in turn.steps)
    return in_tok * in_rate / 1_000_000, out_tok * out_rate / 1_000_000, model


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


def _classify_message(msg: dict) -> list[tuple[str, str, int]]:
    """Return (role, block_type, bytes) for each content block in a message."""
    role = msg["role"]
    content = msg["content"]
    if isinstance(content, str):
        return [(role, "text", len(content))]
    out = []
    for block in content:
        btype = block.get("type", "?") if isinstance(block, dict) else "?"
        size = len(json.dumps(block, default=str))
        out.append((role, btype, size))
    return out


def _request_breakdown(request: dict) -> dict[tuple[str, str], int]:
    """Sum bytes per (role, block_type) across messages + system + tools."""
    cats: dict[tuple[str, str], int] = {}
    for m in request.get("messages", []):
        for role, btype, size in _classify_message(m):
            cats[(role, btype)] = cats.get((role, btype), 0) + size
    system = request.get("system")
    if system:
        size = len(system) if isinstance(system, str) else len(json.dumps(system, default=str))
        cats[("system", "text")] = cats.get(("system", "text"), 0) + size
    tools = request.get("tools")
    if tools:
        cats[("tools", "schema")] = len(json.dumps(tools, default=str))
    return cats


def _body_budget(turn: Turn) -> str:
    """Byte-attribution + approximate cost report for one turn."""
    if not turn.steps:
        return "  (no steps)"
    step_totals = [sum(_request_breakdown(s.request).values()) for s in turn.steps]
    total_sent = sum(step_totals)
    peak = step_totals[-1]
    cats = _request_breakdown(turn.steps[-1].request)
    in_cost, out_cost, model = _turn_cost(turn)
    total_cost = in_cost + out_cost
    lines = [
        f"  model: {model}",
        f"  steps: {len(turn.steps)}",
        f"  bytes: total_sent={_fmt_bytes(total_sent)}"
        f"  peak_request={_fmt_bytes(peak)}"
        f"  redundancy={total_sent / peak:.1f}x",
        f"  cost:  input=${in_cost:.4f}  output=${out_cost:.4f}  total=${total_cost:.4f}",
        "  peak breakdown:",
    ]
    for (role, btype), size in sorted(cats.items(), key=lambda x: -x[1]):
        pct = 100 * size / peak
        cat_cost = in_cost * size / peak  # attribute input $ by peak-share of bytes
        lines.append(
            f"    {role:10s} {btype:12s}  {_fmt_bytes(size):>10s}  {pct:5.1f}%  ${cat_cost:.4f}"
        )
    return "\n".join(lines)


def _body_default(turn: Turn, full: bool = False) -> str:
    """Content-focused body: user input + assistant reply, truncated by default."""
    out = turn.text or f"(no reply — reason={turn.terminal_reason})"
    in_label = paint("In: ", "cyan")
    out_label = paint("Out:", "cyan")
    if full:
        return f"  {in_label} {turn.user_input}\n  {out_label} {out}"
    return f"  {in_label} {_truncate(turn.user_input)}\n  {out_label} {_truncate(out)}"


def render(
    indexed: list[tuple[int, Turn]],
    trace: bool,
    full: bool = False,
    budget: bool = False,
) -> str:
    """Content-focused body by default; --trace for full trace, --budget for byte report.

    Every block is prefixed by a rule header carrying the original
    file-position index, so it stays visible under `--turn N` / `--tail K`.
    """
    blocks = []
    for i, t in indexed:
        if budget:
            body = _body_budget(t)
        elif trace:
            body = t.trace()
        else:
            body = _body_default(t, full=full)
        blocks.append(f"{_header(i)}\n{body}")
    out = "\n\n".join(blocks)
    if budget:
        grand_bytes = sum(
            sum(_request_breakdown(s.request).values()) for _, t in indexed for s in t.steps
        )
        grand_cost = sum(sum(_turn_cost(t)[:2]) for _, t in indexed)
        out += (
            f"\n\nGrand total: {_fmt_bytes(grand_bytes)} sent"
            f", ~${grand_cost:.4f} spent"
        )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="View a gene turn log (JSONL).")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="Path to the JSONL log file. Defaults to the newest file in logs/.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--trace",
        action="store_true",
        help="Print full per-step trace for each turn (default: index / in / out).",
    )
    mode.add_argument(
        "--budget",
        action="store_true",
        help="Report bytes sent per turn, broken down by (role, block_type).",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Show full user input and assistant reply (no truncation, preserve newlines).",
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

    print(render(indexed, trace=args.trace, full=args.no_truncate, budget=args.budget))


if __name__ == "__main__":
    main()
