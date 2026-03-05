#!/usr/bin/env python3
"""Run compatibility matrix tests locally with one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Row:
    python_version: str
    integration: str
    version_set: str


WORKFLOW_ROWS: list[Row] = [
    # compat job
    *[
        Row(py, integration, version_set)
        for py in ("3.10", "3.11", "3.12", "3.13")
        for integration in ("onnx", "torch", "timm", "tensorflow")
        for version_set in ("min", "current")
        if not (py == "3.13" and integration == "tensorflow")
    ],
    # compat-canary-314 job
    *[
        Row("3.14", integration, "current")
        for integration in ("torch", "timm")
    ],
]


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
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=REPO_ROOT)
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
        "--with-editable",
        ".",
        "--with",
        "pytest",
    ]
    for dep in deps:
        cmd.extend(["--with", dep])
    cmd.extend(["python", "-m", "pytest", f"tests/compat/test_{row.integration}_compat.py", "-q"])

    result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=REPO_ROOT)
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
    args = parser.parse_args()

    passed = 0
    failed = 0
    skipped = 0

    for row in WORKFLOW_ROWS:
        label = f"{row.python_version} | {row.integration} | {row.version_set}"
        print(f"==> {label}")
        status, output = run_row(row)
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
