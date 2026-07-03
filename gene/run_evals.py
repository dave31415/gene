"""Matrix runner: run every eval suite against every eval config.

Writes per-cell results to `eval_results/<suite>/<config>.json` and a
summary to `eval_results/summary.md`. Default mode is diff-only: compare
current run to what's on disk and print the delta without touching files.
Pass `--save` to overwrite baseline files.

Timings are excluded from the results file (they belong in a separate
timings log recorded only when caching is disabled — not yet implemented).
"""

import argparse
import json
from pathlib import Path
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


def diff_cell(current: dict, baseline: dict | None) -> list[str]:
    """Return human-readable diff lines. Empty means bit-identical."""
    if baseline is None:
        s = current["summary"]
        return [f"    no baseline yet ({s['passed']}/{s['total']} would be recorded)"]

    lines: list[str] = []
    base_cases = {c["name"]: c for c in baseline["cases"]}
    curr_cases = {c["name"]: c for c in current["cases"]}

    for name in sorted(curr_cases.keys() - base_cases.keys()):
        lines.append(f"    + new case: {name}")
    for name in sorted(base_cases.keys() - curr_cases.keys()):
        lines.append(f"    - removed case: {name}")

    for name in sorted(curr_cases.keys() & base_cases.keys()):
        cur = curr_cases[name]
        base = base_cases[name]
        if cur["passed"] and not base["passed"]:
            lines.append(f"    FIXED   {name}: fail → pass")
        elif not cur["passed"] and base["passed"]:
            lines.append(f"    REGRESS {name}: pass → fail")
        if cur["output_tokens"] != base["output_tokens"]:
            lines.append(
                f"    tokens  {name}: out {base['output_tokens']} → {cur['output_tokens']}"
            )
    return lines


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
            results = run(cases, config=config)
            report = build_report(results, config=config)
            current = cell_to_dict(report, suite=suite, config_name=config_name)
            cells.append(current)

            cell_path = RESULTS_DIR / suite / f"{config_name}.json"
            baseline = None
            if cell_path.exists():
                baseline = json.loads(cell_path.read_text())

            diff_lines = diff_cell(current, baseline)
            if diff_lines:
                changed += 1
                for line in diff_lines:
                    print(line)
            else:
                print("    no changes")

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
        top_summary_path.write_text(
            json.dumps(build_top_summary(cells), indent=2) + "\n"
        )

    mode = "saved to baseline" if args.save else "diff only (no writes)"
    print(f"\n=== done: {changed}/{len(cells)} cells changed | mode: {mode} ===")


if __name__ == "__main__":
    main()
