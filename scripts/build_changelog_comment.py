#!/usr/bin/env python3
"""Generate a changelog preview comment for release PRs using the GitHub API."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import tomllib


def get_version() -> str:
    data = tomllib.loads(Path("pyproject.toml").read_text())
    return data["project"]["version"]


def get_previous_tag(repo: str) -> str | None:
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/releases/latest", "--jq", ".tag_name"],
        capture_output=True,
        text=True,
    )
    tag = result.stdout.strip()
    return tag if result.returncode == 0 and tag else None


def generate_notes(repo: str, tag_name: str, target: str, prev_tag: str | None) -> str:
    args = [
        "gh",
        "api",
        f"repos/{repo}/releases/generate-notes",
        "-f",
        f"tag_name={tag_name}",
        "-f",
        f"target_commitish={target}",
    ]
    if prev_tag:
        args += ["-f", f"previous_tag_name={prev_tag}"]
    args += ["--jq", ".body"]

    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout


def build_comment(tag_name: str, notes: str) -> str:
    return (
        f"## Changelog preview for `{tag_name}`\n\n"
        "> Preview of the GitHub release notes that will be generated when this is tagged.\n\n"
        + notes
    )


def main() -> None:
    repo = os.environ["REPO"]
    head_ref = os.environ["HEAD_REF"]
    output = os.environ.get("OUTPUT", "/tmp/changelog-preview.md")
    tag_name = os.environ.get("TAG_NAME") or f"v{get_version()}"

    prev_tag = get_previous_tag(repo)
    notes = generate_notes(repo, tag_name, head_ref, prev_tag)
    Path(output).write_text(build_comment(tag_name, notes))


if __name__ == "__main__":
    main()
