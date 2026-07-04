"""Eval runner: load a case suite, send each prompt through `CachedAnthropic`,
report pass/fail with token and timing stats.

Case suites live in `gene.agent.eval_cases.<name>` and expose `CASES: list[Case]`.
Run one with `uv run python -m gene.agent.evals --suite <name> [--model <tag>]`.
"""

import argparse
import importlib
import time
from pathlib import Path
from typing import Any

from gene.agent.config import get_llm_config
from gene.agent.eval_case import Case, Report, Result
from gene.agent.llm import CachedAnthropic


def run(
    cases: list[Case],
    config: dict[str, Any],
    use_cache: bool = True,
) -> list[Result]:
    llm = CachedAnthropic(config=config, use_cache=use_cache)
    results = []
    for case in cases:
        t0 = time.perf_counter()
        msg, _ = llm.send(messages=[{"role": "user", "content": case.prompt}])
        elapsed = time.perf_counter() - t0
        results.append(
            Result(
                case=case,
                passed=case.check(msg),
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
                seconds=elapsed,
            )
        )
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
    print(f"{'case':<28} {'result':<6} {'in':>6} {'out':>6} {'sec':>6}")
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        print(
            f"{r.case.name:<28} {mark:<6} "
            f"{r.input_tokens:>6} {r.output_tokens:>6} {r.seconds:>6.2f}"
        )
    print(
        f"\n{report.passed}/{report.total} passed  |  "
        f"{report.total_input_tokens} in / {report.total_output_tokens} out tokens  |  "
        f"{report.total_seconds:.2f}s total"
    )


def load_suite(name: str) -> list[Case]:
    """Dynamically import `gene.agent.eval_cases.<name>` and return its `CASES` list."""
    module = importlib.import_module(f"gene.agent.eval_cases.{name}")
    return module.CASES


def list_suites() -> list[str]:
    """Enumerate every `.py` file under `gene/agent/eval_cases/` (except `__init__`)."""
    pkg_dir = Path(__file__).resolve().parent / "eval_cases"
    return sorted(p.stem for p in pkg_dir.glob("*.py") if p.name != "__init__.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a gene eval suite.")
    parser.add_argument(
        "--suite",
        default="basic",
        help="Suite name under gene.agent.eval_cases (default: basic)",
    )
    parser.add_argument(
        "--model",
        choices=["haiku", "sonnet", "opus"],
        default="sonnet",
        help="Model tag (default: sonnet)",
    )
    args = parser.parse_args()
    cases = load_suite(args.suite)
    config = get_llm_config(model=args.model)
    results = run(cases, config=config)
    report = build_report(results, config=config)
    print_report(report)


if __name__ == "__main__":
    main()
