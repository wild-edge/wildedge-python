#!/usr/bin/env python3
"""Compatibility matrix helper for integration dependency sets."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata

MATRIX = {
    "anthropic": {
        "min": ["anthropic==0.25.0"],
        "current": ["anthropic==1.15.0"],
    },
    "onnx": {
        "min": ["onnxruntime==1.18.1", "numpy==1.26.4"],
        "current": ["onnxruntime==1.31.0", "numpy==2.2.6"],
    },
    "torch": {
        "min": ["torch==2.4.1", "numpy==1.26.4"],
        "current": ["torch==2.14.1", "numpy==2.2.6"],
    },
    "timm": {
        "min": [
            "torch==2.4.1",
            "torchvision==0.19.1",
            "timm==1.0.11",
            "numpy==1.26.4",
        ],
        "current": [
            "torch==2.14.1",
            "torchvision==0.29.1",
            "timm==1.0.30",
            "numpy==2.2.6",
        ],
    },
    "tensorflow": {
        "min": ["tensorflow==2.16.1", "keras==3.3.3", "numpy==1.26.4"],
        "current": ["tensorflow==2.21.0", "keras==3.15.1", "numpy==2.2.6"],
    },
    "llama_cpp": {
        "min": ["llama-cpp-python==0.2.90", "numpy==1.26.4"],
        "current": ["llama-cpp-python==0.3.36", "numpy==2.2.6"],
    },
    "openai": {
        "min": ["openai==1.30.0"],
        "current": ["openai==3.35.1"],
    },
    "transformers": {
        "min": ["transformers==4.40.0", "torch==2.4.1", "numpy==1.26.4"],
        "current": ["transformers==5.20.1", "torch==2.14.1", "numpy==2.2.6"],
    },
    "ultralytics": {
        "min": ["ultralytics==8.0.200", "numpy==1.26.4"],
        "current": ["ultralytics==8.4.182", "numpy==2.2.6"],
    },
    "mlx": {
        "current": ["mlx==0.32.4", "mlx-lm==0.32.2"],
    },
    "faster_whisper": {
        "current": ["faster-whisper==1.2.1", "ctranslate2==4.8.2", "numpy==2.2.6"]
    },
    "mlx_whisper": {"current": ["mlx-whisper==0.4.3", "mlx==0.32.3", "numpy==2.2.6"]},
}

SUPPORTED_PYTHON = {
    "anthropic": ["3.10", "3.11", "3.12", "3.13"],
    "onnx": ["3.10", "3.11", "3.12", "3.13", "3.14"],
    "torch": ["3.10", "3.11", "3.12", "3.13", "3.14"],
    "timm": ["3.10", "3.11", "3.12", "3.13", "3.14"],
    "tensorflow": ["3.10", "3.11", "3.12"],
    "llama_cpp": ["3.10", "3.11", "3.12", "3.13"],
    "openai": ["3.10", "3.11", "3.12", "3.13"],
    "transformers": ["3.10", "3.11", "3.12", "3.13"],
    "ultralytics": ["3.10", "3.11", "3.12", "3.13"],
    "mlx": ["3.12", "3.13"],
    "faster_whisper": ["3.12", "3.13"],
    "mlx_whisper": ["3.12", "3.13"],
}

# Integrations without a row of their own, and the row that exercises them.
COVERED_BY = {
    "keras": "tensorflow",
}

# Integrations that need a specific runner; everything else runs on Linux.
RUNNER = {
    "mlx": "macos-14",
    "mlx_whisper": "macos-14",
}

# Python versions run as non-blocking canaries, and only with current pins.
CANARY_PYTHON = {"3.14"}

# Interpreter-specific overrides where upstream wheels are unavailable for older pins.
PYTHON_DEP_OVERRIDES = {
    "onnx": {
        "min": {
            "3.13": ["onnxruntime==1.20.1", "numpy==2.1.3"],
        },
        "current": {
            "3.10": ["onnxruntime==1.23.2", "numpy==2.2.6"],
        },
    },
    "torch": {
        "min": {
            "3.13": ["torch==2.6.0", "numpy==2.1.3"],
            "3.14": ["torch==2.5.0", "numpy==2.1.3"],
        },
        "current": {
            "3.14": ["torch==2.14.1", "numpy==2.5.4"],
        },
    },
    "tensorflow": {
        "current": {
            "3.10": ["tensorflow==2.21.0", "keras==3.12.4", "numpy==2.2.6"],
        },
    },
    "transformers": {
        "min": {
            "3.13": ["transformers==4.45.0", "torch==2.5.0", "numpy==2.1.3"],
        }
    },
    "timm": {
        "min": {
            "3.13": [
                "torch==2.6.0",
                "torchvision==0.21.0",
                "timm==1.0.11",
                "numpy==2.1.3",
            ],
            "3.14": [
                "torch==2.5.0",
                "torchvision==0.19.1",
                "timm==1.0.11",
                "numpy==2.1.3",
            ],
        },
        "current": {
            "3.14": [
                "torch==2.14.1",
                "torchvision==0.29.1",
                "timm==1.0.30",
                "numpy==2.5.4",
            ],
        },
    },
}


def resolve_deps(integration: str, version_set: str, python_version: str) -> list[str]:
    override = (
        PYTHON_DEP_OVERRIDES.get(integration, {})
        .get(version_set, {})
        .get(python_version)
    )
    if override:
        return override
    return MATRIX[integration][version_set]


def print_deps(integration: str, version_set: str, python_version: str) -> int:
    if integration not in MATRIX:
        print(f"Unknown integration: {integration}", file=sys.stderr)
        return 2
    if version_set not in MATRIX[integration]:
        print(f"Unknown version set: {version_set}", file=sys.stderr)
        return 2
    if python_version not in SUPPORTED_PYTHON[integration]:
        print(
            f"Unsupported combo: integration={integration} python={python_version}",
            file=sys.stderr,
        )
        return 3
    print("\n".join(resolve_deps(integration, version_set, python_version)))
    return 0


def print_rows() -> int:
    rows = [
        {"integration": integration, "version_set": version_set, "python_version": py}
        for integration, sets in MATRIX.items()
        for version_set in sets
        for py in SUPPORTED_PYTHON[integration]
    ]
    print(json.dumps(rows))
    return 0


def print_table() -> int:
    print("| Integration | Version set | Dependencies | Supported Python |")
    print("|---|---|---|---|")
    for integration, sets in MATRIX.items():
        supported = ", ".join(SUPPORTED_PYTHON[integration])
        for version_set, deps in sets.items():
            dep_str = ", ".join(f"`{d}`" for d in deps)
            print(f"| `{integration}` | `{version_set}` | {dep_str} | {supported} |")
            overrides = PYTHON_DEP_OVERRIDES.get(integration, {}).get(version_set, {})
            for py, override_deps in sorted(overrides.items()):
                override_str = ", ".join(f"`{d}`" for d in override_deps)
                print(
                    f"| `{integration}` | `{version_set}` (`py{py}` override) | "
                    f"{override_str} | {py} |"
                )
    return 0


def _strip_local(version: str) -> str:
    return version.split("+", 1)[0]


def verify(integration: str, version_set: str, python_version: str) -> int:
    """Check that the installed distributions match the pins exactly.

    pip can satisfy a pin and then let another package upgrade it, or fall
    back to a different release; this makes the job fail instead.
    """
    mismatches = []
    for requirement in resolve_deps(integration, version_set, python_version):
        name, _, wanted = requirement.partition("==")
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{name}: not installed, pinned {wanted}")
            continue
        if _strip_local(installed) != wanted:
            mismatches.append(f"{name}: installed {installed}, pinned {wanted}")
    for line in mismatches:
        print(line, file=sys.stderr)
    return 1 if mismatches else 0


def github_matrix() -> list[dict]:
    """Rows for the compat workflow: every supported combination.

    Canary Pythons run with current pins only and don't fail the run.
    """
    rows = []
    for integration, sets in MATRIX.items():
        runner = RUNNER.get(integration, "ubuntu-latest")
        for version_set in sets:
            pythons = [
                py
                for py in SUPPORTED_PYTHON[integration]
                if py not in CANARY_PYTHON or version_set == "current"
            ]
            for py in pythons:
                rows.append(
                    {
                        "integration": integration,
                        "version-set": version_set,
                        "python-version": py,
                        "runs-on": runner,
                        "canary": py in CANARY_PYTHON,
                    }
                )
    return rows


def print_github_matrix() -> int:
    print(json.dumps({"include": github_matrix()}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("deps", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--integration", required=True)
        cmd.add_argument("--version-set", required=True, choices=["min", "current"])
        cmd.add_argument("--python-version", required=True)

    sub.add_parser("rows")
    sub.add_parser("table")
    sub.add_parser("github-matrix")

    args = parser.parse_args()
    if args.cmd == "deps":
        return print_deps(args.integration, args.version_set, args.python_version)
    if args.cmd == "verify":
        return verify(args.integration, args.version_set, args.python_version)
    if args.cmd == "rows":
        return print_rows()
    if args.cmd == "table":
        return print_table()
    if args.cmd == "github-matrix":
        return print_github_matrix()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
