"""Keep scripts/compat_matrix.py, tests/compat and the docs in step."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from wildedge.integrations.registry import INTEGRATION_SPECS

ROOT = Path(__file__).resolve().parents[1]
COMPAT_TESTS = ROOT / "tests" / "compat"

_spec = importlib.util.spec_from_file_location(
    "compat_matrix", ROOT / "scripts" / "compat_matrix.py"
)
compat_matrix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compat_matrix)


def test_every_integration_has_a_compat_row():
    for spec in INTEGRATION_SPECS:
        row = compat_matrix.COVERED_BY.get(spec.name, spec.name)
        assert row in compat_matrix.MATRIX, f"{spec.name} has no compat row"


def test_covered_by_points_at_real_rows():
    for integration, row in compat_matrix.COVERED_BY.items():
        assert integration not in compat_matrix.MATRIX
        assert row in compat_matrix.MATRIX


def test_every_row_has_a_test_file_and_supported_pythons():
    assert set(compat_matrix.SUPPORTED_PYTHON) == set(compat_matrix.MATRIX)
    for integration in compat_matrix.MATRIX:
        assert (COMPAT_TESTS / f"test_{integration}_compat.py").exists()


def test_no_compat_test_file_without_a_row():
    for path in COMPAT_TESTS.glob("test_*_compat.py"):
        integration = path.name[len("test_") : -len("_compat.py")]
        assert integration in compat_matrix.MATRIX, path.name


def test_compat_tests_never_skip():
    # A skip would turn a broken install into a green job.
    for path in COMPAT_TESTS.glob("*.py"):
        source = path.read_text()
        assert "importorskip" not in source, path.name
        assert not re.search(r"pytest\.skip\(|mark\.skip", source), path.name


def test_pins_are_exact():
    pin = re.compile(r"^[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*$")
    for integration, sets in compat_matrix.MATRIX.items():
        for deps in sets.values():
            assert all(pin.match(d) for d in deps), (integration, deps)
    for integration, by_set in compat_matrix.PYTHON_DEP_OVERRIDES.items():
        for version_set, by_python in by_set.items():
            assert version_set in compat_matrix.MATRIX[integration]
            for py, deps in by_python.items():
                assert py in compat_matrix.SUPPORTED_PYTHON[integration]
                assert all(pin.match(d) for d in deps), (integration, py, deps)


def test_docs_table_matches_the_matrix(capsys):
    compat_matrix.print_table()
    table = capsys.readouterr().out.strip()
    docs = (ROOT / "docs" / "compatibility.md").read_text()
    assert table in docs, (
        "docs/compatibility.md is out of date; regenerate it with "
        "`python scripts/compat_matrix.py table`"
    )


def test_pr_profile_runs_min_on_oldest_and_current_on_newest():
    rows = compat_matrix.github_matrix("pr")
    seen = {(r["integration"], r["version-set"]): r for r in rows}
    for integration, sets in compat_matrix.MATRIX.items():
        if compat_matrix.RUNNER.get(integration, "ubuntu-latest") != "ubuntu-latest":
            assert not any(r["integration"] == integration for r in rows)
            continue
        stable = [
            py
            for py in compat_matrix.SUPPORTED_PYTHON[integration]
            if py not in compat_matrix.CANARY_PYTHON
        ]
        for version_set in sets:
            row = seen[(integration, version_set)]
            expected = stable[0] if version_set == "min" else stable[-1]
            assert row["python-version"] == expected
    assert len(rows) == len(seen)
    assert not any(r["canary"] for r in rows)


def test_full_profile_covers_every_stable_combination():
    rows = compat_matrix.github_matrix("full")
    keys = {(r["integration"], r["version-set"], r["python-version"]) for r in rows}
    assert len(keys) == len(rows)
    for integration, sets in compat_matrix.MATRIX.items():
        for version_set in sets:
            for py in compat_matrix.SUPPORTED_PYTHON[integration]:
                canary = py in compat_matrix.CANARY_PYTHON
                if canary and version_set != "current":
                    assert (integration, version_set, py) not in keys
                else:
                    assert (integration, version_set, py) in keys
    for row in rows:
        assert row["canary"] == (row["python-version"] in compat_matrix.CANARY_PYTHON)
        expected_runner = compat_matrix.RUNNER.get(row["integration"], "ubuntu-latest")
        assert row["runs-on"] == expected_runner


@pytest.fixture
def installed(monkeypatch):
    versions: dict[str, str] = {}

    def version(name):
        if name not in versions:
            raise compat_matrix.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(compat_matrix.metadata, "version", version)
    return versions


def test_verify_accepts_matching_versions(installed, capsys):
    installed.update({"onnxruntime": "1.31.0", "numpy": "2.2.6"})
    assert compat_matrix.verify("onnx", "current", "3.12") == 0
    assert capsys.readouterr().err == ""


def test_verify_ignores_local_version_suffix(installed):
    installed.update({"torch": "2.14.1+cpu", "numpy": "2.2.6"})
    assert compat_matrix.verify("torch", "current", "3.12") == 0


def test_verify_reports_a_different_version(installed, capsys):
    installed.update({"onnxruntime": "1.31.0", "numpy": "2.3.0"})
    assert compat_matrix.verify("onnx", "current", "3.12") == 1
    assert "numpy: installed 2.3.0, pinned 2.2.6" in capsys.readouterr().err


def test_verify_reports_a_missing_package(installed, capsys):
    installed.update({"numpy": "2.2.6"})
    assert compat_matrix.verify("onnx", "current", "3.12") == 1
    assert "onnxruntime: not installed" in capsys.readouterr().err


def test_verify_uses_python_overrides(installed):
    installed.update({"onnxruntime": "1.23.2", "numpy": "2.2.6"})
    assert compat_matrix.verify("onnx", "current", "3.10") == 0
