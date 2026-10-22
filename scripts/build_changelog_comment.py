#!/usr/bin/env python3
"""Generate changelog text for release PRs and for tag-build release notes.

The notes are the version's section of CHANGELOG.md, so they list what changed
for users rather than every commit and merge since the previous tag.

The two callers differ only in wording: release-pr.yml renders a preview of
notes that do not exist yet, while release.yml (which sets TAG_NAME) renders
the notes themselves.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

CHANGELOG = Path("CHANGELOG.md")


def get_version() -> str:
    # Regex instead of tomllib so the script also runs on Python 3.10.
    match = re.search(
        r'^version = "([^"]+)"', Path("pyproject.toml").read_text(), re.MULTILINE
    )
    if match is None:
        raise SystemExit("could not find project version in pyproject.toml")
    return match.group(1)


def get_previous_tag() -> str | None:
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v*", "HEAD^"],
        capture_output=True,
        text=True,
    )
    tag = result.stdout.strip()
    return tag if result.returncode == 0 and tag else None


def changelog_section(text: str, version: str) -> str:
    """Body of the ``## <version>`` section, up to the next ``## `` heading."""
    heading = re.compile(rf"^## \[?{re.escape(version)}\]?(?:\s|$)", re.MULTILINE)
    match = heading.search(text)
    if match is None:
        raise SystemExit(f"CHANGELOG.md has no section for {version}")
    line_end = text.find("\n", match.start())
    start = len(text) if line_end == -1 else line_end + 1
    end = re.compile(r"^## ", re.MULTILINE).search(text, start)
    body = text[start : end.start() if end else len(text)].strip()
    if not body:
        raise SystemExit(f"CHANGELOG.md section for {version} is empty")
    return body


def build_comment(
    tag_name: str,
    notes: str,
    prev_tag: str | None,
    *,
    preview: bool = True,
) -> str:
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
        f"{notes}\n\n"
        f"**Full Changelog**: {range_str}\n"
    )


def main() -> None:
    output = os.environ.get("OUTPUT", "/tmp/changelog-preview.md")
    # release.yml sets TAG_NAME on tag builds and release-pr.yml never does. Its
    # absence is what distinguishes a PR preview from the real release notes.
    tag_name = os.environ.get("TAG_NAME")
    preview = not tag_name
    tag_name = tag_name or f"v{get_version()}"

    notes = changelog_section(CHANGELOG.read_text(), tag_name.removeprefix("v"))
    prev_tag = get_previous_tag()
    Path(output).write_text(build_comment(tag_name, notes, prev_tag, preview=preview))


if __name__ == "__main__":
    main()
