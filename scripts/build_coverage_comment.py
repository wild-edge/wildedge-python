#!/usr/bin/env python3
"""Build a markdown coverage summary from coverage.py JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_comment(coverage: dict) -> str:
    totals = coverage["totals"]
    pct = totals["percent_covered_display"]
    covered = totals["covered_lines"]
    stmts = totals["num_statements"]

    files = []
    for path, data in coverage["files"].items():
        summary = data["summary"]
        missed = summary["num_statements"] - summary["covered_lines"]
        if missed > 0 and not path.startswith("tests/"):
            files.append((missed, summary["percent_covered"], path))
    files.sort(reverse=True)
    top = files[:5]

    lines = [
        "## Coverage report",
        "",
        f"- Total: **{pct}%** ({covered}/{stmts} lines)",
        "",
        "### Top uncovered files",
    ]
    if top:
        for missed, percent, path in top:
            lines.append(f"- `{path}`: {percent:.1f}% ({missed} lines missed)")
    else:
        lines.append("- No uncovered source files.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-json",
        default="coverage.json",
        help="Path to coverage.py json output",
    )
    parser.add_argument(
        "--output",
        default="coverage-comment.md",
        help="Path to output markdown file",
    )
    args = parser.parse_args()

    coverage = json.loads(Path(args.coverage_json).read_text())
    comment = build_comment(coverage)
    Path(args.output).write_text(comment)


if __name__ == "__main__":
    main()
