#!/usr/bin/env python3
"""Run compatibility matrix tests locally with one command."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Row:
    python_version: str
    integration: str
    version_set: str


def load_rows() -> list[Row]:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "compat_matrix.py"), "rows"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return [Row(**r) for r in json.loads(result.stdout)]


WORKFLOW_ROWS: list[Row] = load_rows()


UNSUPPORTED_MARKERS = (
    "No solution found when resolving",
    "No matching distribution found",
    "has no wheels with a matching",
)


def get_deps(row: Row) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "compat_matrix.py"),
        "deps",
        "--integration",
        row.integration,
        "--version-set",
        row.version_set,
        "--python-version",
        row.python_version,
    ]
    result = subprocess.run(
        cmd, check=False, capture_output=True, text=True, cwd=REPO_ROOT
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def run_row(row: Row) -> tuple[str, str]:
    deps = get_deps(row)
    cmd = [
        "uv",
        "run",
        "--python",
        row.python_version,
        "--link-mode=copy",
        "--with-editable",
        ".",
        "--with",
        "pytest",
        "--with",
        "pytest-asyncio",
        "--with",
        "pytest-mock",
    ]
    for dep in deps:
        cmd.extend(["--with", dep])
    cmd.extend(
        [
            "python",
            "-m",
            "pytest",
            f"tests/compat/test_{row.integration}_compat.py",
            "-q",
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        env = {**os.environ, "UV_PROJECT_ENVIRONMENT": tmpdir}
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, cwd=REPO_ROOT, env=env
        )
    output = f"{result.stdout}\n{result.stderr}".strip()

    if result.returncode == 0:
        return "PASS", output
    if any(marker in output for marker in UNSUPPORTED_MARKERS):
        return "SKIP_UNSUPPORTED", output
    return "FAIL", output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-unsupported",
        action="store_true",
        help="Treat unsupported dependency rows as failures.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of rows to run in parallel (default: cpu count).",
    )
    args = parser.parse_args()

    passed = 0
    failed = 0
    skipped = 0

    futures = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for row in WORKFLOW_ROWS:
            futures[executor.submit(run_row, row)] = row

        for future in as_completed(futures):
            row = futures[future]
            label = f"{row.python_version} | {row.integration} | {row.version_set}"
            status, output = future.result()
            print(f"==> {label}")
            print(status)
            if output:
                print(output)
            print()

            if status == "PASS":
                passed += 1
            elif status == "SKIP_UNSUPPORTED":
                skipped += 1
                if args.strict_unsupported:
                    failed += 1
            else:
                failed += 1

    print(
        "SUMMARY "
        f"passed={passed} failed={failed} skipped_unsupported={skipped} total={len(WORKFLOW_ROWS)}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
