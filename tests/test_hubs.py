"""Tests for wildedge.hubs hub trackers."""

from __future__ import annotations

import os
import types
from unittest.mock import patch

from wildedge.hubs.huggingface import HuggingFaceHubTracker
from wildedge.hubs.torchhub import TorchHubTracker

# ---------------------------------------------------------------------------
# BaseHubTracker.scan_cache
# ---------------------------------------------------------------------------


def test_scan_cache_returns_real_files_skips_symlinks(tmp_path):
    real_file = tmp_path / "blob"
    real_file.write_bytes(b"x" * 100)
    link = tmp_path / "link"
    link.symlink_to(real_file)

    tracker = HuggingFaceHubTracker()
    with patch.object(tracker, "cache_dir", return_value=str(tmp_path)):
        result = tracker.scan_cache()

    assert str(real_file) in result
    assert str(link) not in result
    assert result[str(real_file)] == 100


def test_scan_cache_returns_empty_when_no_cache_dir():
    tracker = HuggingFaceHubTracker()
    with patch.object(tracker, "cache_dir", return_value=None):
        assert tracker.scan_cache() == {}


def test_scan_cache_returns_empty_when_dir_missing():
    tracker = HuggingFaceHubTracker()
    with patch.object(tracker, "cache_dir", return_value="/nonexistent/path/xyz"):
        assert tracker.scan_cache() == {}


# ---------------------------------------------------------------------------
# HuggingFaceHubTracker.diff_to_records
# ---------------------------------------------------------------------------


def test_hf_diff_to_records_groups_by_repo():
    tracker = HuggingFaceHubTracker()
    sep = os.sep
    before = {}
    after = {
        f"{sep}cache{sep}hub{sep}models--facebook--opt-125m{sep}blobs{sep}sha1": 200_000_000,
        f"{sep}cache{sep}hub{sep}models--facebook--opt-125m{sep}snapshots{sep}abc{sep}config.json": 1_000,
        f"{sep}cache{sep}hub{sep}models--bert-base-uncased{sep}blobs{sep}sha2": 400_000_000,
    }
    records = tracker.diff_to_records(before, after, elapsed_ms=5000)

    assert len(records) == 2
    repo_ids = {r["repo_id"] for r in records}
    assert repo_ids == {"facebook/opt-125m", "bert-base-uncased"}
    for r in records:
        assert r["source_type"] == "huggingface"
        assert r["source_url"] == f"hf://{r['repo_id']}"
        assert r["cache_hit"] is False
        assert r["duration_ms"] == 5000

    opt = next(r for r in records if r["repo_id"] == "facebook/opt-125m")
    assert opt["size"] == 200_000_000 + 1_000


def test_hf_diff_to_records_returns_empty_when_no_new_files():
    tracker = HuggingFaceHubTracker()
    snapshot = {"/cache/blobs/sha1": 100}
    assert tracker.diff_to_records(snapshot, snapshot, elapsed_ms=1000) == []


def test_hf_diff_to_records_ignores_files_outside_models_dirs():
    tracker = HuggingFaceHubTracker()
    before = {}
    after = {"/cache/hub/some_other_file.txt": 500}
    # Files not under a models-- directory are silently dropped (no repo_id)
    records = tracker.diff_to_records(before, after, elapsed_ms=1000)
    assert records == []


# ---------------------------------------------------------------------------
# TorchHubTracker.diff_to_records
# ---------------------------------------------------------------------------


def test_torch_hub_diff_to_records_checkpoints():
    tracker = TorchHubTracker()
    hub_dir = "/home/user/.cache/torch/hub"
    before = {}
    after = {f"{hub_dir}/checkpoints/resnet50-0676ba61.pth": 97_781_926}

    with patch.object(tracker, "cache_dir", return_value=hub_dir):
        records = tracker.diff_to_records(before, after, elapsed_ms=3000)

    assert len(records) == 1
    r = records[0]
    assert r["source_type"] == "torchhub"
    assert r["source_url"] == "torchhub://checkpoints/resnet50-0676ba61.pth"
    assert r["repo_id"] == "resnet50.pth"  # hash suffix stripped
    assert r["cache_hit"] is False
    assert r["size"] == 97_781_926


def test_torch_hub_diff_to_records_repo_clone_dir():
    tracker = TorchHubTracker()
    hub_dir = "/home/user/.cache/torch/hub"
    before = {}
    after = {
        f"{hub_dir}/pytorch_vision_v0.10.0/hubconf.py": 2_000,
        f"{hub_dir}/pytorch_vision_v0.10.0/torchvision/models/resnet.py": 30_000,
    }

    with patch.object(tracker, "cache_dir", return_value=hub_dir):
        records = tracker.diff_to_records(before, after, elapsed_ms=2000)

    assert len(records) == 2
    for r in records:
        assert r["source_type"] == "torchhub"
        assert r["source_url"] == "torchhub://pytorch/vision"
        assert r["repo_id"] == "pytorch/vision"


def test_torch_hub_diff_to_records_empty_when_no_new_files():
    tracker = TorchHubTracker()
    snapshot = {"/cache/torch/hub/checkpoints/model.pth": 1000}
    with patch.object(tracker, "cache_dir", return_value="/cache/torch/hub"):
        assert tracker.diff_to_records(snapshot, snapshot, elapsed_ms=500) == []


# ---------------------------------------------------------------------------
# TorchHubTracker.install_patch idempotency
# ---------------------------------------------------------------------------


def test_torch_hub_install_patch_is_idempotent(monkeypatch):
    import wildedge.hubs.torchhub as torchhub_mod

    original_load_calls = []

    class FakeHub:
        @staticmethod
        def load(repo_or_dir, model, *args, **kwargs):
            original_load_calls.append((repo_or_dir, model))
            return object()

        @staticmethod
        def get_dir():
            return "/tmp/hub"

    fake_torch = types.SimpleNamespace(hub=FakeHub)
    monkeypatch.setattr(torchhub_mod, "_torch", fake_torch)
    monkeypatch.setattr(torchhub_mod, "_torch_hub_load_patched", False)

    tracker = TorchHubTracker()
    tracker.install_patch(lambda: None)
    first_patched = fake_torch.hub.load
    tracker.install_patch(lambda: None)
    second_patched = fake_torch.hub.load

    assert first_patched is second_patched
    assert getattr(first_patched, "__wildedge_patch_name__", None) == "torchhub_load"


# ---------------------------------------------------------------------------
# HuggingFaceHubTracker.drain (thread-local buffer)
# ---------------------------------------------------------------------------


def test_hf_tracker_drain_returns_and_clears_buffer(monkeypatch):
    import wildedge.hubs.huggingface as hf_mod

    tracker = HuggingFaceHubTracker()
    # Directly inject a record into the thread-local buffer
    hf_mod._buffer().append({"repo_id": "test/model", "source_type": "huggingface"})

    result = tracker.drain()
    assert len(result) == 1
    assert result[0]["repo_id"] == "test/model"
    # Buffer should be cleared
    assert tracker.drain() == []
