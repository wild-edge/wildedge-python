#!/usr/bin/env python3
"""Validate that a Git tag version matches pyproject.toml project version."""

from __future__ import annotations

import os
import re
from pathlib import Path


def main() -> None:
    tag = os.environ["TAG_NAME"]
    if not tag.startswith("v"):
        raise SystemExit(f"Expected tag to start with 'v', got: {tag}")
    tag_version = tag[1:]

    # Regex instead of tomllib so the script also runs on Python 3.10.
    match = re.search(
        r'^version = "([^"]+)"', Path("pyproject.toml").read_text(), re.MULTILINE
    )
    if match is None:
        raise SystemExit("could not find project version in pyproject.toml")
    project_version = match.group(1)

    if tag_version != project_version:
        raise SystemExit(
            f"Tag version '{tag_version}' does not match project version "
            f"'{project_version}' from pyproject.toml"
        )
    print(f"Validated tag version: {tag_version}")


if __name__ == "__main__":
    main()
