"""Eval runner: load a suite of cases, execute each against an LLM (one-shot
or full agent loop), report pass/fail with token and timing stats.

Suites live in two roots:
- `gene.agent.eval_cases.<name>` — core suites (name is just `<name>`)
- `gene.genealogy.eval_cases.<name>` — genealogy suites (name is
  `genealogy/<name>`)

Names are stable across the two roots — the `<domain>/` prefix picks the
package. A suite module exports `CASES` and, for genealogy, `TAG` naming
the family DB it talks to. Suites are skipped cleanly when the DB isn't
built.

Run one with `uv run python -m gene.agent.evals --suite <name> [--model <tag>]`.
"""

import argparse
import importlib
import time
from pathlib import Path
from typing import Any, NamedTuple

from gene.agent.config import get_llm_config
from gene.agent.eval_case import Case, Report, Result, TurnCase
from gene.agent.llm import CachedAnthropic
from gene.genealogy.agent import build_conversation
from gene.genealogy.config import get_db_path

# TODO(domain-separation): the core runner shouldn't hard-import genealogy.
#   Two things to think through together:
#     1. `_run_turn` / `skip_reason` reach into `gene.genealogy.*` directly.
#        Consider a domain-registration hook so genealogy (and future
#        domains) plug into the runner without the runner knowing them.
#     2. Suite names with a `/` translate directly into eval_results paths
#        (`eval_results/genealogy/david_ancestors/<config>.json`). Happy
#        accident today, but couples name format to on-disk layout — worth
#        deciding whether that's the intended contract before it hardens.
_GENEALOGY_PREFIX = "genealogy/"


class Suite(NamedTuple):
    """A loaded suite: its qualified name, cases, and optional family tag."""

    name: str
    cases: list[Case | TurnCase]
    tag: str | None


def _resolve_module(name: str) -> str:
    """Map a qualified suite name to its dotted module path."""
    if name.startswith(_GENEALOGY_PREFIX):
        return f"gene.genealogy.eval_cases.{name.removeprefix(_GENEALOGY_PREFIX)}"
    return f"gene.agent.eval_cases.{name}"


def load_suite(name: str) -> Suite:
    """Import the suite module for `name` and wrap its exports in a Suite."""
    module = importlib.import_module(_resolve_module(name))
    return Suite(name=name, cases=module.CASES, tag=getattr(module, "TAG", None))


def skip_reason(suite: Suite) -> str | None:
    """None if runnable; a short human-readable reason otherwise."""
    if suite.tag is None:
        return None
    if not get_db_path(suite.tag).exists():
        return f"data '{suite.tag}' not built (run: python -m gene.genealogy.load {suite.tag})"
    return None


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


def _run_turn(case: TurnCase, tag: str, llm: CachedAnthropic) -> Result:
    t0 = time.perf_counter()
    conv = build_conversation(tag, llm=llm)
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
    """Execute every case in the suite; caller must ensure the suite is runnable."""
    llm = CachedAnthropic(config=config, use_cache=use_cache)
    results: list[Result] = []
    for case in suite.cases:
        if isinstance(case, TurnCase):
            results.append(_run_turn(case, suite.tag, llm))
        else:
            results.append(_run_one_shot(case, llm))
    return results


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


def list_suites() -> list[str]:
    """Enumerate every suite across both roots, with `genealogy/` prefix where relevant."""
    root = Path(__file__).resolve().parent.parent
    core = sorted(
        p.stem
        for p in (root / "agent" / "eval_cases").glob("*.py")
        if p.name != "__init__.py"
    )
    genealogy = sorted(
        f"genealogy/{p.stem}"
        for p in (root / "genealogy" / "eval_cases").glob("*.py")
        if p.name != "__init__.py"
    )
    return core + genealogy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a gene eval suite.")
    parser.add_argument(
        "--suite",
        default="basic",
        help="Qualified suite name, e.g. 'basic' or 'genealogy/david_ancestors'.",
    )
    parser.add_argument(
        "--model",
        choices=["haiku", "sonnet", "opus"],
        default="sonnet",
        help="Model tag (default: sonnet)",
    )
    args = parser.parse_args()

    suite = load_suite(args.suite)
    reason = skip_reason(suite)
    if reason is not None:
        print(f"SKIP: suite {suite.name} — {reason}")
        return

    config = get_llm_config(model=args.model)
    results = run(suite, config=config)
    report = build_report(results, config=config)
    print_report(report)


if __name__ == "__main__":
    main()
