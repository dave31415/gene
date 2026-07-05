"""Eval runner: load a suite of cases from a directory, execute each against an
LLM (one-shot or full agent loop), report pass/fail with token and timing stats.

The runner is domain-agnostic. You point it at a directory of suite modules
and it discovers what's there:

    uv run python -m gene.agent.evals gene/agent/eval_cases
    uv run python -m gene.agent.evals gene/genealogy/eval_cases
    uv run python -m gene.agent.evals gene/genealogy/eval_cases --suite david_ancestors
    uv run python -m gene.agent.evals gene/genealogy/eval_cases --suite david_ancestors --name davids_parents

Suite modules export:
- `CASES: list[Case | TurnCase]` — required
- `build_conversation(llm) -> Conversation` — required if any case is a TurnCase
- `precheck() -> str | None` — optional; return a skip reason (e.g. "DB not built")

The runner reads these by duck-typed convention. No import from any use-case
package appears in this module — that's the whole point of the reshape.
"""

import argparse
import importlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from gene.agent.config import get_llm_config
from gene.agent.conversation import Conversation
from gene.agent.eval_case import Case, Report, Result, TurnCase
from gene.agent.llm import CachedAnthropic


class Suite(NamedTuple):
    """A loaded suite: name, cases, and the optional callables the module exports."""

    name: str
    cases: list[Case | TurnCase]
    build_conversation: Callable[[CachedAnthropic], Conversation] | None
    precheck: Callable[[], str | None] | None


def _dir_to_package(path: str | Path) -> str:
    """Convert a POSIX-style repo-relative dir to a dotted module path."""
    return str(Path(path)).replace("/", ".").rstrip(".")


def list_suites(dir_path: str | Path) -> list[str]:
    """Every `.py` file (bar `__init__`) in `dir_path`, by stem, sorted."""
    return sorted(
        p.stem
        for p in Path(dir_path).glob("*.py")
        if p.name != "__init__.py"
    )


def load_suite(dir_path: str | Path, name: str) -> Suite:
    """Import `<dir>.<name>` and wrap its exports in a Suite."""
    module = importlib.import_module(f"{_dir_to_package(dir_path)}.{name}")
    return Suite(
        name=name,
        cases=module.CASES,
        build_conversation=getattr(module, "build_conversation", None),
        precheck=getattr(module, "precheck", None),
    )


def skip_reason(suite: Suite) -> str | None:
    """None if runnable; a short human-readable reason otherwise."""
    if suite.precheck is None:
        return None
    return suite.precheck()


def _run_one_shot(case: Case, llm: CachedAnthropic) -> Result:
    t0 = time.perf_counter()
    msg, _ = llm.send(messages=[{"role": "user", "content": case.prompt}])
    elapsed = time.perf_counter() - t0
    return Result(
        case=case,
        passed=case.check(msg),
        input_tokens=msg.usage.input_tokens,
        output_tokens=msg.usage.output_tokens,
        seconds=elapsed,
        steps=1,
        tool_calls=0,
    )


def _run_turn(case: TurnCase, conv: Conversation) -> Result:
    t0 = time.perf_counter()
    turn = conv.ask(case.prompt)
    elapsed = time.perf_counter() - t0
    tool_calls = sum(len(s.tool_calls) for s in turn.steps)
    return Result(
        case=case,
        passed=case.check(turn),
        input_tokens=turn.input_tokens,
        output_tokens=turn.output_tokens,
        seconds=elapsed,
        steps=len(turn.steps),
        tool_calls=tool_calls,
    )


def run(suite: Suite, config: dict[str, Any], use_cache: bool = True) -> list[Result]:
    """Execute every case in the suite; caller must ensure the suite is runnable.

    A fresh `Conversation` is built per TurnCase so cases don't share history.
    """
    llm = CachedAnthropic(config=config, use_cache=use_cache)
    results: list[Result] = []
    for case in suite.cases:
        if isinstance(case, TurnCase):
            if suite.build_conversation is None:
                raise ValueError(
                    f"suite {suite.name!r} has TurnCase {case.name!r} but no "
                    "build_conversation(llm) function"
                )
            conv = suite.build_conversation(llm)
            results.append(_run_turn(case, conv))
        else:
            results.append(_run_one_shot(case, llm))
    return results


def filter_cases(suite: Suite, name: str | None) -> Suite:
    """Return a new Suite keeping only cases whose name matches (or all if None)."""
    if name is None:
        return suite
    kept = [c for c in suite.cases if c.name == name]
    if not kept:
        raise ValueError(f"no case named {name!r} in suite {suite.name!r}")
    return suite._replace(cases=kept)


def build_report(results: list[Result], config: dict[str, Any]) -> Report:
    return Report(
        model=config["model"],
        results=results,
        passed=sum(1 for r in results if r.passed),
        total=len(results),
        total_input_tokens=sum(r.input_tokens for r in results),
        total_output_tokens=sum(r.output_tokens for r in results),
        total_seconds=sum(r.seconds for r in results),
    )


def print_report(report: Report) -> None:
    print(f"\n=== eval report (model={report.model}) ===")
    print(f"{'case':<32} {'result':<6} {'in':>6} {'out':>6} {'steps':>5} {'tools':>5} {'sec':>6}")
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        print(
            f"{r.case.name:<32} {mark:<6} "
            f"{r.input_tokens:>6} {r.output_tokens:>6} "
            f"{r.steps:>5} {r.tool_calls:>5} {r.seconds:>6.2f}"
        )
    print(
        f"\n{report.passed}/{report.total} passed  |  "
        f"{report.total_input_tokens} in / {report.total_output_tokens} out tokens  |  "
        f"{report.total_seconds:.2f}s total"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gene.agent.evals",
        description="Run eval suites in a directory against an LLM.",
    )
    parser.add_argument(
        "dir",
        help="Directory of suite modules (e.g. gene/agent/eval_cases).",
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="Suite (file stem) to run. Default: every suite in the directory.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Single case name inside the suite. Requires --suite.",
    )
    parser.add_argument(
        "--model",
        choices=["haiku", "sonnet", "opus"],
        default="sonnet",
        help="Model tag (default: sonnet)",
    )
    args = parser.parse_args()

    if args.name is not None and args.suite is None:
        parser.error("--name requires --suite")

    suite_names = [args.suite] if args.suite else list_suites(args.dir)
    if not suite_names:
        print(f"no suites found in {args.dir}")
        return

    config = get_llm_config(model=args.model)

    for name in suite_names:
        suite = load_suite(args.dir, name)
        reason = skip_reason(suite)
        if reason is not None:
            print(f"\nSKIP: suite {suite.name} — {reason}")
            continue
        suite = filter_cases(suite, args.name)
        results = run(suite, config=config)
        report = build_report(results, config=config)
        print(f"\n>> suite: {suite.name}")
        print_report(report)


if __name__ == "__main__":
    main()
