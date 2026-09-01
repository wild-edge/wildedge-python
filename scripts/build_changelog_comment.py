#!/usr/bin/env python3
"""Generate changelog text for release PRs and for tag-build release notes.

The two callers differ only in wording: release-pr.yml renders a preview of
notes that do not exist yet, while release.yml (which sets TAG_NAME) renders
the notes themselves.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def get_version() -> str:
    # Regex instead of tomllib so the script also runs on Python 3.10.
    match = re.search(
        r'^version = "([^"]+)"', Path("pyproject.toml").read_text(), re.MULTILINE
    )
    if match is None:
        raise SystemExit("could not find project version in pyproject.toml")
    return match.group(1)


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


def build_comment(
    tag_name: str,
    commits: list[str],
    prev_tag: str | None,
    *,
    preview: bool = True,
) -> str:
    items = "\n".join(f"- {c}" for c in commits) if commits else "- No changes."
    range_str = f"{prev_tag}...{tag_name}" if prev_tag else tag_name
    heading = "Changelog preview for" if preview else "Changelog for"
    blurb = (
        "> Preview of the release notes that will be generated when this is tagged.\n\n"
        if preview
        else ""
    )
    return (
        f"## {heading} `{tag_name}`\n\n"
        f"{blurb}"
        f"**What's Changed**\n\n{items}\n\n"
        f"**Full Changelog**: {range_str}\n"
    )


def main() -> None:
    output = os.environ.get("OUTPUT", "/tmp/changelog-preview.md")
    # release.yml sets TAG_NAME on tag builds; release-pr.yml never does, so its
    # absence is what distinguishes a PR preview from the real release notes.
    tag_name = os.environ.get("TAG_NAME")
    preview = not tag_name
    tag_name = tag_name or f"v{get_version()}"

    prev_tag = get_previous_tag()
    commits = get_commits_since(prev_tag)
    Path(output).write_text(build_comment(tag_name, commits, prev_tag, preview=preview))


if __name__ == "__main__":
    main()
