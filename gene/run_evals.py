"""Matrix runner: run every eval suite against every eval config.

Writes per-cell results to `eval_results/<suite>/<config>.json` plus
per-suite (`<suite>/summary.json`) and cross-suite (`summary.json`)
roll-ups. Default mode is diff-only: compare the current run to what's
on disk and print the delta without touching files. Pass `--save` to
overwrite baseline files.

Timings are excluded from the results files. Passing `--no-cache`
bypasses the LLM cache and appends one row per run to
`<suite>/<config>.timings.jsonl`, then rebuilds `timings_summary.json`
files (per-suite and top-level) from the accumulated history.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from gene.eval_case import Report
from gene.eval_configs import get_eval_configs
from gene.evals import build_report, list_suites, load_suite, run

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results"


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
    """Aggregate timing across all suites, keyed by config. `mean_total_seconds`
    is the sum of per-suite mean totals — average duration of one full matrix run."""
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


def build_top_summary(all_cells: list[dict]) -> dict[str, Any]:
    """Aggregate every cell across suites, keyed by config. Answers the
    cross-model comparison question ("how does haiku do overall?")."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the eval matrix.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Overwrite baseline files (default: diff-only, no writes).",
    )
    parser.add_argument("--suite", help="Limit to a single suite.")
    parser.add_argument("--config", help="Limit to a single config.")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the LLM cache and append timings to each cell's .timings.jsonl.",
    )
    args = parser.parse_args()

    suites = [args.suite] if args.suite else list_suites()
    configs = get_eval_configs()
    if args.config:
        configs = {args.config: configs[args.config]}

    cells: list[dict] = []
    changed = 0

    for suite in suites:
        cases = load_suite(suite)
        for config_name, config in configs.items():
            print(f"\n>> {suite} × {config_name} ({config['model']})")
            results = run(cases, config=config, use_cache=not args.no_cache)
            report = build_report(results, config=config)
            current = cell_to_dict(report, suite=suite, config_name=config_name)
            cells.append(current)

            cell_path = RESULTS_DIR / suite / f"{config_name}.json"
            baseline = None
            if cell_path.exists():
                baseline = json.loads(cell_path.read_text())

            status, diffs = diff_cell(current, baseline)
            print(status)
            for line in diffs:
                print(line)
            if baseline is None or diffs:
                changed += 1

            if args.no_cache:
                timings_path = RESULTS_DIR / suite / f"{config_name}.timings.jsonl"
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
        for suite in suites:
            suite_cells = [c for c in cells if c["suite"] == suite]
            suite_summary_path = RESULTS_DIR / suite / "summary.json"
            suite_summary_path.parent.mkdir(parents=True, exist_ok=True)
            suite_summary_path.write_text(
                json.dumps(build_suite_summary(suite_cells), indent=2) + "\n"
            )

        top_summary_path = RESULTS_DIR / "summary.json"
        top_summary_path.parent.mkdir(parents=True, exist_ok=True)
        top_summary_path.write_text(json.dumps(build_top_summary(cells), indent=2) + "\n")

    if args.no_cache:
        write_timing_summaries(RESULTS_DIR)

    mode = "saved to baseline" if args.save else "diff only (no writes)"
    print(f"\n=== done: {changed}/{len(cells)} cells changed | mode: {mode} ===")


if __name__ == "__main__":
    main()
