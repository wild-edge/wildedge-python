"""The download helper the compat tests use for test models."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "compat_conftest", Path(__file__).parent / "compat" / "conftest.py"
)
compat_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compat_conftest)

URL = "https://example.invalid/model.bin"
DATA = b"tiny model"
SHA = hashlib.sha256(DATA).hexdigest()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(compat_conftest, "DOWNLOAD_CACHE", tmp_path)
    return tmp_path


@pytest.fixture
def downloads(monkeypatch):
    calls: list[str] = []
    served = {"data": DATA, "error": None}

    def urlretrieve(url, filename):
        calls.append(url)
        if served["error"] is not None:
            raise served["error"]
        Path(filename).write_bytes(served["data"])

    monkeypatch.setattr(compat_conftest.urllib.request, "urlretrieve", urlretrieve)
    return calls, served


def test_downloads_and_verifies(cache, downloads):
    calls, _ = downloads
    path = compat_conftest.downloaded_fixture("model.bin", URL, SHA)
    assert path == cache / "model.bin"
    assert path.read_bytes() == DATA
    assert calls == [URL]
    assert not (cache / "model.bin.part").exists()


def test_reuses_a_cached_file(cache, downloads):
    calls, _ = downloads
    (cache / "model.bin").write_bytes(DATA)
    compat_conftest.downloaded_fixture("model.bin", URL, SHA)
    assert calls == []


def test_replaces_a_corrupt_cached_file(cache, downloads):
    calls, _ = downloads
    (cache / "model.bin").write_bytes(b"truncated")
    path = compat_conftest.downloaded_fixture("model.bin", URL, SHA)
    assert path.read_bytes() == DATA
    assert calls == [URL]


def test_wrong_checksum_fails_and_keeps_nothing(cache, downloads):
    _, served = downloads
    served["data"] = b"something else"
    with pytest.raises(pytest.fail.Exception, match="expected " + SHA):
        compat_conftest.downloaded_fixture("model.bin", URL, SHA)
    assert list(cache.iterdir()) == []


def test_download_error_fails_with_the_url(cache, downloads):
    _, served = downloads
    served["error"] = OSError("404 Not Found")
    with pytest.raises(pytest.fail.Exception, match=URL):
        compat_conftest.downloaded_fixture("model.bin", URL, SHA)
