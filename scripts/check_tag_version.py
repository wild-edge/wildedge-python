#!/usr/bin/env python3
"""Validate that a Git tag version matches pyproject.toml project version."""

from __future__ import annotations

import os
from pathlib import Path

import tomllib


def main() -> None:
    tag = os.environ["TAG_NAME"]
    if not tag.startswith("v"):
        raise SystemExit(f"Expected tag to start with 'v', got: {tag}")
    tag_version = tag[1:]

    data = tomllib.loads(Path("pyproject.toml").read_text())
    project_version = data["project"]["version"]

    if tag_version != project_version:
        raise SystemExit(
            f"Tag version '{tag_version}' does not match project version "
            f"'{project_version}' from pyproject.toml"
        )
    print(f"Validated tag version: {tag_version}")


if __name__ == "__main__":
    main()
