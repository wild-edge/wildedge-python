from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "build_changelog_comment.py"

COMMITS = ["Add a thing (#1)", "Fix another thing (#2)"]


def load_script():
    spec = importlib.util.spec_from_file_location("build_changelog_comment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preview_wording_on_pull_requests():
    module = load_script()
    text = module.build_comment("v1.2.3", COMMITS, "v1.2.2", preview=True)

    assert text.startswith("## Changelog preview for `v1.2.3`")
    assert "> Preview of the release notes" in text
    assert "**Full Changelog**: v1.2.2...v1.2.3" in text
    for commit in COMMITS:
        assert f"- {commit}" in text


def test_release_notes_drop_the_preview_wording():
    """Tag builds render the notes themselves, not a preview of them."""
    module = load_script()
    text = module.build_comment("v1.2.3", COMMITS, "v1.2.2", preview=False)

    assert text.startswith("## Changelog for `v1.2.3`")
    assert "preview" not in text.lower()
    assert "**Full Changelog**: v1.2.2...v1.2.3" in text
    for commit in COMMITS:
        assert f"- {commit}" in text


def test_first_release_falls_back_to_the_tag_alone():
    module = load_script()
    text = module.build_comment("v0.1.0", COMMITS, None, preview=False)

    assert "**Full Changelog**: v0.1.0" in text


def test_empty_commit_range_is_reported():
    module = load_script()
    text = module.build_comment("v1.2.3", [], "v1.2.2", preview=False)

    assert "- No changes." in text


def test_release_commits_are_filtered_from_the_log(monkeypatch):
    module = load_script()
    log = "Release 1.2.3\nAdd a thing (#1)\n\nBump version to 1.2.3\n"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"stdout": log, "returncode": 0})(),
    )

    assert module.get_commits_since("v1.2.2") == [
        "Add a thing (#1)",
        "Bump version to 1.2.3",
    ]


def test_get_version_reads_pyproject_without_tomllib():
    module = load_script()
    assert module.get_version().count(".") >= 1
