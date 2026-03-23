#!/usr/bin/env python3
"""Generate a changelog preview comment for release PRs using the GitHub API."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import tomllib


def get_version() -> str:
    data = tomllib.loads(Path("pyproject.toml").read_text())
    return data["project"]["version"]


RELEASE_COMMIT_RE = re.compile(r"^Release \d+\.\d+\.\d+$")


def get_previous_tag() -> str | None:
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v*", "HEAD^"],
        capture_output=True,
        text=True,
    )
    tag = result.stdout.strip()
    return tag if result.returncode == 0 and tag else None


def get_commits_since(prev_tag: str | None) -> list[str]:
    revision = f"{prev_tag}..HEAD" if prev_tag else "HEAD"
    result = subprocess.run(
        ["git", "log", revision, "--pretty=format:%s"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return [line for line in lines if not RELEASE_COMMIT_RE.match(line)]


def build_comment(tag_name: str, commits: list[str], prev_tag: str | None) -> str:
    items = "\n".join(f"- {c}" for c in commits) if commits else "- No changes."
    range_str = f"{prev_tag}...{tag_name}" if prev_tag else tag_name
    return (
        f"## Changelog preview for `{tag_name}`\n\n"
        "> Preview of the release notes that will be generated when this is tagged.\n\n"
        f"**What's Changed**\n\n{items}\n\n"
        f"**Full Changelog**: {range_str}\n"
    )


def main() -> None:
    output = os.environ.get("OUTPUT", "/tmp/changelog-preview.md")
    tag_name = os.environ.get("TAG_NAME") or f"v{get_version()}"

    prev_tag = get_previous_tag()
    commits = get_commits_since(prev_tag)
    Path(output).write_text(build_comment(tag_name, commits, prev_tag))


if __name__ == "__main__":
    main()
