#!/usr/bin/env python3
"""Build llms.txt (index) and llms-full.txt (full content) from README + docs.

llms.txt is a link index per https://llmstxt.org; llms-full.txt is the entire
documentation concatenated and stamped with the release version, so an agent
gets correct, versioned docs in one fetch. Run at release time; outputs land
in OUTPUT_DIR (default: llms-dist).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).parent.parent

# Order is the reading order in llms-full.txt.
SOURCES: list[tuple[str, str, str]] = [
    ("README.md", "", "WildEdge Python SDK"),
    ("docs/configuration.md", "configuration", "Configuration"),
    ("docs/deployment.md", "deployment", "Deployment"),
    ("docs/manual-tracking.md", "manual-tracking", "Manual tracking"),
    ("docs/llm_api.md", "llm_api", "LLM API tracking"),
    ("docs/compatibility.md", "compatibility", "Compatibility matrix"),
]


def project_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return data["project"]["version"]


def clean_markdown(text: str) -> str:
    """Drop HTML tags and badge lines; they are noise to a text consumer."""
    kept = [
        line for line in text.splitlines() if not line.lstrip().startswith(("<", "[!["))
    ]
    return "\n".join(kept).strip() + "\n"


def first_paragraph(text: str) -> str:
    """First prose paragraph outside code fences, as the index description."""
    collected: list[str] = []
    in_fence = False
    for line in clean_markdown(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith(("#", ">", "|", "-")):
            if collected:
                break
            continue
        if not stripped:
            if collected:
                break
            continue
        collected.append(stripped)
    paragraph = " ".join(collected)
    return paragraph if len(paragraph) <= 220 else paragraph[:217] + "..."


def build(output_dir: Path, base_url: str) -> None:
    version = project_version()
    stamp = (
        f"wildedge-sdk {version} documentation, generated "
        f"{datetime.now(timezone.utc).date().isoformat()} from the v{version} release."
    )

    index_lines = [
        "# WildEdge Python SDK",
        "",
        f"> {stamp}",
        "",
        "On-device ML inference monitoring for Python.",
        "",
        "## Docs",
        "",
    ]
    full_parts = [f"# WildEdge Python SDK ({version})\n\n> {stamp}\n"]

    for relpath, slug, title in SOURCES:
        path = REPO_ROOT / relpath
        text = path.read_text(encoding="utf-8")
        url = f"{base_url}/{slug}" if slug else base_url
        description = first_paragraph(text)
        index_lines.append(f"- [{title}]({url}): {description}")
        full_parts.append(f"\n---\n\n{clean_markdown(text)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "llms.txt").write_text(
        "\n".join(index_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "llms-full.txt").write_text("".join(full_parts), encoding="utf-8")
    print(f"wrote {output_dir / 'llms.txt'} and {output_dir / 'llms-full.txt'}")


def main() -> None:
    output_dir = Path(os.environ.get("OUTPUT_DIR", "llms-dist"))
    base_url = os.environ.get("BASE_URL", "https://docs.wildedge.dev").rstrip("/")
    build(output_dir, base_url)


if __name__ == "__main__":
    main()
