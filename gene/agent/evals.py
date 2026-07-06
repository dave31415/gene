"""Eval runner: load suites from a directory, execute each case against every
named config, diff against saved baselines, and (optionally) print a per-case
table or refresh baselines / timings.

The runner is domain-agnostic. Point it at a directory of suite modules and it
discovers what's there:

    uv run python -m gene.agent.evals gene/agent/eval_cases
    uv run python -m gene.agent.evals gene/genealogy/eval_cases --config sonnet
    uv run python -m gene.agent.evals gene/genealogy/eval_cases --suite david_ancestors
    uv run python -m gene.agent.evals gene/genealogy/eval_cases --suite david_ancestors --name davids_parents
    uv run python -m gene.agent.evals gene/agent/eval_cases --save                  # overwrite baselines
    uv run python -m gene.agent.evals gene/agent/eval_cases --no-cache              # real API calls + timings
    uv run python -m gene.agent.evals gene/agent/eval_cases -v                      # per-case table

Suite modules export:
- `CASES: list[Case | TurnCase]` — required
- `build_conversation(llm) -> Conversation` — required if any case is a TurnCase
- `precheck() -> str | None` — optional; return a skip reason (e.g. "DB not built")

The runner reads these by duck-typed convention. No import from any use-case
package appears in this module.

Results mirror the input dir: `eval_results/<dir>/<suite>/<config>.json` plus
per-suite (`<suite>/summary.json`) and cross-suite (`summary.json`) roll-ups.
Default mode is diff-only. `--save` overwrites baseline files. Timings are
excluded from baseline JSON — `--no-cache` bypasses the LLM cache, records
one row per run to `<suite>/<config>.timings.jsonl`, and rebuilds
`timings_summary.json` files from the accumulated history.
"""

import argparse
import importlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, NamedTuple

from gene.agent.conversation import Conversation
from gene.agent.eval_case import Case, Report, Result, TurnCase
from gene.agent.eval_configs import get_eval_configs
from gene.agent.llm import CachedAnthropic

RESULTS_ROOT = Path("eval_results")


class Suite(NamedTuple):
    """A loaded suite: name, cases, and the optional callables the module exports."""

    name: str
    cases: list[Case | TurnCase]
    build_conversation: Callable[[CachedAnthropic], Conversation] | None
    precheck: Callable[[], str | None] | None


# ---------- suite loading ----------


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


def filter_cases(suite: Suite, name: str | None) -> Suite:
    """Return a new Suite keeping only cases whose name matches (or all if None)."""
    if name is None:
        return suite
    kept = [c for c in suite.cases if c.name == name]
    if not kept:
        raise ValueError(f"no case named {name!r} in suite {suite.name!r}")
    return suite._replace(cases=kept)


# ---------- case execution ----------


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


# ---------- baseline serialization + diff ----------


def cell_to_dict(report: Report, suite: str, config_name: str) -> dict[str, Any]:
    """Serialize a run cell for baseline storage. Excludes timings and paths."""
    return {
        "suite": suite,
        "config_name": config_name,
        "model": report.model,
        "cases": [
            {
                "name": r.case.name,
                "passed": r.passed,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "steps": r.steps,
                "tool_calls": r.tool_calls,
            }
            for r in report.results
        ],
        "summary": {
            "passed": report.passed,
            "total": report.total,
            "total_input_tokens": report.total_input_tokens,
            "total_output_tokens": report.total_output_tokens,
        },
    }


def diff_cell(current: dict, baseline: dict | None) -> tuple[str, list[str]]:
    """Return (status, diffs). `status` is always shown ("X/Y passed" or
    "X/Y passed (was A/B)"). `diffs` lists per-case flips and token deltas;
    empty means nothing meaningful changed."""
    s = current["summary"]
    curr_pass = f"{s['passed']}/{s['total']}"

    if baseline is None:
        return f"    {curr_pass} passed (no baseline)", []

    base_s = baseline["summary"]
    base_pass = f"{base_s['passed']}/{base_s['total']}"
    status = (
        f"    {curr_pass} passed"
        if curr_pass == base_pass
        else f"    {curr_pass} passed (was {base_pass})"
    )

    diffs: list[str] = []
    base_cases = {c["name"]: c for c in baseline["cases"]}
    curr_cases = {c["name"]: c for c in current["cases"]}

    for name in sorted(curr_cases.keys() - base_cases.keys()):
        diffs.append(f"    + new case: {name}")
    for name in sorted(base_cases.keys() - curr_cases.keys()):
        diffs.append(f"    - removed case: {name}")

    for name in sorted(curr_cases.keys() & base_cases.keys()):
        cur = curr_cases[name]
        base = base_cases[name]
        if cur["passed"] and not base["passed"]:
            diffs.append(f"    FIXED   {name}: fail → pass")
        elif not cur["passed"] and base["passed"]:
            diffs.append(f"    REGRESS {name}: pass → fail")
        if cur["output_tokens"] != base["output_tokens"]:
            diffs.append(
                f"    tokens  {name}: out {base['output_tokens']} → {cur['output_tokens']}"
            )
        if cur.get("steps", 1) != base.get("steps", 1):
            diffs.append(f"    steps   {name}: {base.get('steps')} → {cur['steps']}")
        if cur.get("tool_calls", 0) != base.get("tool_calls", 0):
            diffs.append(
                f"    tools   {name}: {base.get('tool_calls')} → {cur['tool_calls']}"
            )
    return status, diffs


def build_suite_summary(cells: list[dict]) -> dict[str, Any]:
    """Roll up every cell of a single suite into a config-keyed dict."""
    return {
        cell["config_name"]: {
            "model": cell["model"],
            "passed": cell["summary"]["passed"],
            "total": cell["summary"]["total"],
            "total_input_tokens": cell["summary"]["total_input_tokens"],
            "total_output_tokens": cell["summary"]["total_output_tokens"],
        }
        for cell in cells
    }


def build_top_summary(all_cells: list[dict]) -> dict[str, Any]:
    """Aggregate every cell across suites, keyed by config."""
    configs = sorted({c["config_name"] for c in all_cells})
    top: dict[str, Any] = {}
    for config_name in configs:
        cells = [c for c in all_cells if c["config_name"] == config_name]
        top[config_name] = {
            "model": cells[0]["model"],
            "passed": sum(c["summary"]["passed"] for c in cells),
            "total": sum(c["summary"]["total"] for c in cells),
            "total_input_tokens": sum(c["summary"]["total_input_tokens"] for c in cells),
            "total_output_tokens": sum(c["summary"]["total_output_tokens"] for c in cells),
        }
    return top


# ---------- timings ----------


def _stats(samples: list[float]) -> dict[str, Any]:
    """Summary stats over a list of timing samples. Empty→count=0 only."""
    if not samples:
        return {"count": 0}
    return {
        "count": len(samples),
        "mean": round(mean(samples), 3),
        "stddev": round(stdev(samples), 3) if len(samples) > 1 else 0.0,
        "min": round(min(samples), 3),
        "max": round(max(samples), 3),
    }


def build_config_timing_summary(rows: list[dict]) -> dict[str, Any]:
    """Roll up one config's timing history (list of JSONL rows) into stats."""
    case_names: set[str] = set()
    for r in rows:
        case_names.update(r["cases"].keys())
    return {
        "runs": len(rows),
        "cases": {
            name: _stats([r["cases"][name] for r in rows if name in r["cases"]])
            for name in sorted(case_names)
        },
        "total_seconds": _stats([r["total_seconds"] for r in rows]),
    }


def build_suite_timing_summary(suite_dir: Path) -> dict[str, Any]:
    """Read every `<config>.timings.jsonl` in a suite dir, keyed by config."""
    result: dict[str, Any] = {}
    for path in sorted(suite_dir.glob("*.timings.jsonl")):
        config_name = path.name.removesuffix(".timings.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if rows:
            result[config_name] = build_config_timing_summary(rows)
    return result


def build_top_timing_summary(results_dir: Path) -> dict[str, Any]:
    """Aggregate timing across all suites, keyed by config."""
    per_suite: dict[str, dict] = {}
    for suite_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        suite_summary = build_suite_timing_summary(suite_dir)
        if suite_summary:
            per_suite[suite_dir.name] = suite_summary

    configs: set[str] = set()
    for suite_configs in per_suite.values():
        configs.update(suite_configs.keys())

    top: dict[str, Any] = {}
    for config in sorted(configs):
        suite_means = {
            suite: cfgs[config]["total_seconds"]["mean"]
            for suite, cfgs in per_suite.items()
            if config in cfgs
        }
        top[config] = {
            "mean_total_seconds": round(sum(suite_means.values()), 3),
            "per_suite_mean": suite_means,
        }
    return top


def write_timing_summaries(results_dir: Path) -> None:
    """Rebuild every timing summary from the JSONL files on disk."""
    for suite_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if not list(suite_dir.glob("*.timings.jsonl")):
            continue
        suite_summary = build_suite_timing_summary(suite_dir)
        (suite_dir / "timings_summary.json").write_text(json.dumps(suite_summary, indent=2) + "\n")
    top = build_top_timing_summary(results_dir)
    if top:
        (results_dir / "timings_summary.json").write_text(json.dumps(top, indent=2) + "\n")


# ---------- CLI ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gene.agent.evals",
        description="Run eval suites in a directory against every named config; "
                    "diff against saved baselines.",
    )
    parser.add_argument(
        "dir",
        help="Directory of suite modules (e.g. gene/agent/eval_cases).",
    )
    parser.add_argument("--suite", help="Limit to a single suite (file stem).")
    parser.add_argument(
        "--name",
        help="Limit to a single case name inside --suite. Requires --suite.",
    )
    parser.add_argument(
        "--config",
        help="Limit to a single config (default: every config).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Overwrite baseline files (default: diff-only, no writes).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the LLM cache and append timings to each cell's .timings.jsonl.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Also print the per-case report table for each cell.",
    )
    args = parser.parse_args()

    if args.name is not None and args.suite is None:
        parser.error("--name requires --suite")

    results_dir = RESULTS_ROOT / Path(args.dir)

    suite_names = [args.suite] if args.suite else list_suites(args.dir)
    if not suite_names:
        print(f"no suites found in {args.dir}")
        return

    configs = get_eval_configs()
    if args.config:
        configs = {args.config: configs[args.config]}

    cells: list[dict] = []
    changed = 0

    for suite_name in suite_names:
        suite = load_suite(args.dir, suite_name)
        reason = skip_reason(suite)
        if reason is not None:
            print(f"\n>> {suite_name} — SKIP: {reason}")
            continue
        suite = filter_cases(suite, args.name)
        for config_name, config in configs.items():
            print(f"\n>> {suite_name} × {config_name} ({config['model']})")
            results = run(suite, config=config, use_cache=not args.no_cache)
            report = build_report(results, config=config)
            current = cell_to_dict(report, suite=suite_name, config_name=config_name)
            cells.append(current)

            cell_path = results_dir / suite_name / f"{config_name}.json"
            baseline = None
            if cell_path.exists():
                baseline = json.loads(cell_path.read_text())

            status, diffs = diff_cell(current, baseline)
            print(status)
            for line in diffs:
                print(line)
            if args.verbose:
                print_report(report)
            if baseline is None or diffs:
                changed += 1

            if args.no_cache:
                timings_path = results_dir / suite_name / f"{config_name}.timings.jsonl"
                timings_path.parent.mkdir(parents=True, exist_ok=True)
                row = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "cases": {r.case.name: round(r.seconds, 3) for r in report.results},
                    "total_seconds": round(report.total_seconds, 3),
                }
                with timings_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"    timings: {report.total_seconds:.2f}s total (recorded)")

            if args.save:
                cell_path.parent.mkdir(parents=True, exist_ok=True)
                cell_path.write_text(json.dumps(current, indent=2) + "\n")

    if args.save:
        for suite_name in suite_names:
            suite_cells = [c for c in cells if c["suite"] == suite_name]
            if not suite_cells:
                continue
            suite_summary_path = results_dir / suite_name / "summary.json"
            suite_summary_path.parent.mkdir(parents=True, exist_ok=True)
            suite_summary_path.write_text(
                json.dumps(build_suite_summary(suite_cells), indent=2) + "\n"
            )

        if cells:
            top_summary_path = results_dir / "summary.json"
            top_summary_path.parent.mkdir(parents=True, exist_ok=True)
            top_summary_path.write_text(json.dumps(build_top_summary(cells), indent=2) + "\n")

    if args.no_cache and results_dir.exists():
        write_timing_summaries(results_dir)

    mode = "saved to baseline" if args.save else "diff only (no writes)"
    print(f"\n=== done: {changed}/{len(cells)} cells changed | mode: {mode} ===")


if __name__ == "__main__":
    main()
