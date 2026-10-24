from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "build_changelog_comment.py"

CHANGELOG = """# Changelog

Intro text.

## Unreleased

- Not released yet.

## 1.2.3 - 2026-10-21

### Added

- A thing.

### Fixed

- Another thing.

## 1.2.2 - 2026-09-30

### Fixed

- An older thing.
"""


def load_script():
    spec = importlib.util.spec_from_file_location("build_changelog_comment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notes_are_the_version_section_of_the_changelog():
    module = load_script()
    notes = module.changelog_section(CHANGELOG, "1.2.3")

    assert notes == "### Added\n\n- A thing.\n\n### Fixed\n\n- Another thing."


def test_last_section_runs_to_the_end_of_the_file():
    module = load_script()
    assert (
        module.changelog_section(CHANGELOG, "1.2.2") == "### Fixed\n\n- An older thing."
    )


def test_bracketed_version_headings_are_found():
    module = load_script()
    text = "## [2.0.0] - 2026-11-01\n\n- New.\n"
    assert module.changelog_section(text, "2.0.0") == "- New."


def test_version_prefix_does_not_match_a_longer_version():
    module = load_script()
    with pytest.raises(SystemExit, match=r"no section for 1\.2$"):
        module.changelog_section(CHANGELOG, "1.2")


def test_missing_or_empty_section_fails_the_build():
    module = load_script()
    with pytest.raises(SystemExit, match=r"no section for 9\.9\.9"):
        module.changelog_section(CHANGELOG, "9.9.9")
    with pytest.raises(SystemExit, match="is empty"):
        module.changelog_section("## 1.0.0\n\n## 0.9.0\n\n- Old.\n", "1.0.0")


def test_preview_wording_on_pull_requests():
    module = load_script()
    text = module.build_comment("v1.2.3", "- A thing.", "v1.2.2", preview=True)

    assert text.startswith("## Changelog preview for `v1.2.3`")
    assert "> Preview of the release notes" in text
    assert "- A thing." in text
    assert "**Full Changelog**: v1.2.2...v1.2.3" in text


def test_release_notes_drop_the_preview_wording():
    """Tag builds render the notes themselves, not a preview of them."""
    module = load_script()
    text = module.build_comment("v1.2.3", "- A thing.", "v1.2.2", preview=False)

    assert text.startswith("## Changelog for `v1.2.3`")
    assert "preview" not in text.lower()
    assert "**Full Changelog**: v1.2.2...v1.2.3" in text


def test_first_release_falls_back_to_the_tag_alone():
    module = load_script()
    text = module.build_comment("v0.1.0", "- First.", None, preview=False)

    assert "**Full Changelog**: v0.1.0" in text


def test_get_version_reads_pyproject_without_tomllib():
    module = load_script()
    assert module.get_version().count(".") >= 1


def test_current_version_has_a_changelog_section():
    """Catches a release whose CHANGELOG.md entries were left under Unreleased."""
    module = load_script()
    changelog = (SCRIPT.parent.parent / "CHANGELOG.md").read_text()
    assert module.changelog_section(changelog, module.get_version())
