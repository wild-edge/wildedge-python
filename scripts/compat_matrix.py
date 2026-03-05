#!/usr/bin/env python3
"""Compatibility matrix helper for integration dependency sets."""

from __future__ import annotations

import argparse
import sys

MATRIX = {
    "onnx": {
        "min": ["onnxruntime==1.18.1", "numpy==1.26.4"],
        "current": ["onnxruntime==1.20.1", "numpy==2.1.3"],
    },
    "torch": {
        "min": ["torch==2.4.1", "numpy==1.26.4"],
        "current": ["torch==2.6.0", "numpy==2.1.3"],
    },
    "timm": {
        "min": [
            "torch==2.4.1",
            "torchvision==0.19.1",
            "timm==1.0.11",
            "numpy==1.26.4",
        ],
        "current": [
            "torch==2.6.0",
            "torchvision==0.21.0",
            "timm==1.0.15",
            "numpy==2.1.3",
        ],
    },
    "tensorflow": {
        "min": ["tensorflow==2.16.1", "keras==3.3.3", "numpy==1.26.4"],
        "current": ["tensorflow==2.18.0", "keras==3.8.0", "numpy==2.0.2"],
    },
}

SUPPORTED_PYTHON = {
    "onnx": ["3.10", "3.11", "3.12", "3.13", "3.14"],
    "torch": ["3.10", "3.11", "3.12", "3.13", "3.14"],
    "timm": ["3.10", "3.11", "3.12", "3.13", "3.14"],
    "tensorflow": ["3.10", "3.11", "3.12"],
}

# Interpreter-specific overrides where upstream wheels are unavailable for older pins.
PYTHON_DEP_OVERRIDES = {
    "onnx": {
        "min": {
            "3.13": ["onnxruntime==1.20.1", "numpy==2.1.3"],
        }
    },
    "torch": {
        "min": {
            "3.13": ["torch==2.6.0", "numpy==2.1.3"],
            "3.14": ["torch==2.5.0", "numpy==2.1.3"],
        },
        "current": {
            "3.14": ["torch==2.10.0", "numpy==2.1.3"],
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
                "torch==2.10.0",
                "torchvision==0.25.0",
                "timm==1.0.15",
                "numpy==2.1.3",
            ],
        }
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


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    deps = sub.add_parser("deps")
    deps.add_argument("--integration", required=True)
    deps.add_argument("--version-set", required=True, choices=["min", "current"])
    deps.add_argument("--python-version", required=True)

    sub.add_parser("table")

    args = parser.parse_args()
    if args.cmd == "deps":
        return print_deps(args.integration, args.version_set, args.python_version)
    if args.cmd == "table":
        return print_table()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
