from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "build_llms_txt.py"


def load_script():
    spec = importlib.util.spec_from_file_location("build_llms_txt", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_produces_index_and_full_text(tmp_path):
    module = load_script()
    module.build(tmp_path, "https://docs.wildedge.dev")

    index = (tmp_path / "llms.txt").read_text()
    full = (tmp_path / "llms-full.txt").read_text()
    version = module.project_version()

    assert index.startswith("# WildEdge Python SDK")
    assert f"wildedge-sdk {version} documentation" in index
    assert "- [WildEdge Python SDK](https://docs.wildedge.dev):" in index
    for slug in ("configuration", "deployment", "manual-tracking", "llm_api"):
        assert f"https://docs.wildedge.dev/{slug}" in index

    assert f"# WildEdge Python SDK ({version})" in full
    # One recognizable phrase per source document proves each was included.
    assert "Full reference for all `WildEdge` client parameters" in full
    assert "pre-deploy" in full or "wildedge run" in full
    assert "register_model" in full
    assert "llm_api" in full


def test_clean_markdown_strips_html_and_badges():
    module = load_script()
    text = '<p align="center">logo</p>\n[![CI](x)](y)\n# Title\nprose line\n'
    cleaned = module.clean_markdown(text)
    assert "<p" not in cleaned
    assert "[![" not in cleaned
    assert "# Title" in cleaned
    assert "prose line" in cleaned


def test_first_paragraph_skips_headings_and_tables():
    module = load_script()
    text = "# Title\n\n> quote\n\n| a | b |\n\nActual description here.\n"
    assert module.first_paragraph(text) == "Actual description here."


def test_first_paragraph_joins_wrapped_lines_and_skips_fences():
    module = load_script()
    text = "# T\n\n```python\n# comment in code\nx = 1\n```\n\nFirst line\nsecond line.\n\nNext para.\n"
    assert module.first_paragraph(text) == "First line second line."
