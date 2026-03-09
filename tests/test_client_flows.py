from __future__ import annotations

from unittest.mock import patch

import pytest


class DummyModel:
    pass


def test_register_model_fallback_requires_id_when_no_extractor(
    client_with_stubbed_runtime,
):
    client = client_with_stubbed_runtime
    with patch.object(client, "_find_extractor", return_value=None):
        with pytest.raises(ValueError, match="Could not auto-derive a stable model_id"):
            client.register_model(object())


def test_on_model_auto_loaded_uses_hub_records_when_downloads_missing(
    client_with_stubbed_runtime, dummy_handle
):
    client = client_with_stubbed_runtime

    records = [
        {
            "repo_id": "org/model",
            "size": 10,
            "duration_ms": 2,
            "cache_hit": False,
            "bandwidth_bps": 100,
            "source_type": "huggingface",
            "source_url": "hf://org/model",
        }
    ]

    with (
        patch.object(client, "_drain_hub_trackers", return_value=records),
        patch.object(client, "register_model", return_value=dummy_handle),
    ):
        client._on_model_auto_loaded(DummyModel(), load_ms=5)

    dummy_handle.track_download.assert_called_once()
    dummy_handle.track_load.assert_called_once()


def test_on_model_auto_loaded_groups_downloads_by_repo(
    client_with_stubbed_runtime, dummy_handle
):
    client = client_with_stubbed_runtime
    downloads = [
        {
            "repo_id": "org/a",
            "size": 10,
            "duration_ms": 2,
            "cache_hit": False,
            "bandwidth_bps": 100,
            "source_type": "huggingface",
            "source_url": "hf://org/a",
        },
        {
            "repo_id": "org/a",
            "size": 20,
            "duration_ms": 3,
            "cache_hit": True,
            "bandwidth_bps": 200,
            "source_type": "huggingface",
            "source_url": "hf://org/a",
        },
        {
            "repo_id": "org/b",
            "size": 5,
            "duration_ms": 1,
            "cache_hit": True,
            "bandwidth_bps": None,
            "source_type": "huggingface",
            "source_url": "hf://org/b",
        },
    ]

    with patch.object(client, "register_model", return_value=dummy_handle):
        client._on_model_auto_loaded(DummyModel(), load_ms=5, downloads=downloads)

    assert dummy_handle.track_download.call_count == 2
    first_call_kwargs = dummy_handle.track_download.call_args_list[0].kwargs
    assert first_call_kwargs["file_size_bytes"] in (30, 5)
    assert dummy_handle.track_load.called


def test_close_sets_closed_and_closes_consumer(client_with_stubbed_runtime):
    client = client_with_stubbed_runtime
    client.close()
    assert client.closed is True
    client.consumer.close.assert_called_once()


def test_load_skips_duplicate_track_load_for_auto_loaded_model(
    client_with_stubbed_runtime, dummy_handle
):
    client = client_with_stubbed_runtime
    dummy_handle.model_id = "dup-model"
    client.auto_loaded.add("dup-model")

    with patch.object(client, "register_model", return_value=dummy_handle):
        client.load(DummyModel)

    dummy_handle.track_load.assert_not_called()
