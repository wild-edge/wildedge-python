"""Tests for WildEdge client configuration validation."""

from unittest.mock import patch

import pytest

from wildedge import constants


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all dependencies to isolate validation tests."""
    with (
        patch("wildedge.client.detect_device"),
        patch("wildedge.client.Transmitter"),
        patch("wildedge.client.Consumer"),
        patch("wildedge.client.EventQueue"),
        patch("wildedge.client.DeadLetterStore"),
        patch("wildedge.client.ModelRegistry"),
    ):
        yield


def test_batch_size_too_low():
    from wildedge.client import WildEdge

    with pytest.raises(
        ValueError,
        match=f"batch_size must be between {constants.BATCH_SIZE_MIN} and {constants.BATCH_SIZE_MAX}",
    ):
        WildEdge(dsn="https://test@test.com/key", batch_size=0)


def test_batch_size_too_high():
    from wildedge.client import WildEdge

    with pytest.raises(
        ValueError,
        match=f"batch_size must be between {constants.BATCH_SIZE_MIN} and {constants.BATCH_SIZE_MAX}",
    ):
        WildEdge(dsn="https://test@test.com/key", batch_size=101)


def test_flush_interval_too_low():
    from wildedge.client import WildEdge

    with pytest.raises(
        ValueError,
        match=f"flush_interval_sec must be between {constants.FLUSH_INTERVAL_MIN} and {constants.FLUSH_INTERVAL_MAX}",
    ):
        WildEdge(dsn="https://test@test.com/key", flush_interval_sec=0)


def test_flush_interval_too_high():
    from wildedge.client import WildEdge

    with pytest.raises(
        ValueError,
        match=f"flush_interval_sec must be between {constants.FLUSH_INTERVAL_MIN} and {constants.FLUSH_INTERVAL_MAX}",
    ):
        WildEdge(dsn="https://test@test.com/key", flush_interval_sec=3601)


def test_max_queue_size_too_low():
    from wildedge.client import WildEdge

    with pytest.raises(
        ValueError,
        match=f"max_queue_size must be between {constants.MAX_QUEUE_SIZE_MIN} and {constants.MAX_QUEUE_SIZE_MAX}",
    ):
        WildEdge(dsn="https://test@test.com/key", max_queue_size=9)


def test_max_queue_size_too_high():
    from wildedge.client import WildEdge

    with pytest.raises(
        ValueError,
        match=f"max_queue_size must be between {constants.MAX_QUEUE_SIZE_MIN} and {constants.MAX_QUEUE_SIZE_MAX}",
    ):
        WildEdge(dsn="https://test@test.com/key", max_queue_size=10001)


def test_valid_values():
    from wildedge.client import WildEdge

    # Should not raise
    client = WildEdge(
        dsn="https://test@test.com/key",
        batch_size=50,
        flush_interval_sec=120,
        max_queue_size=500,
    )
    assert client is not None


def test_max_event_age_must_be_positive():
    from wildedge.client import WildEdge

    with pytest.raises(ValueError, match="max_event_age_sec must be greater than 0"):
        WildEdge(dsn="https://test@test.com/key", max_event_age_sec=0)


def test_max_dead_letter_batches_must_be_non_negative():
    from wildedge.client import WildEdge

    with pytest.raises(ValueError, match="max_dead_letter_batches must be >= 0"):
        WildEdge(dsn="https://test@test.com/key", max_dead_letter_batches=-1)


def test_app_identity_defaults_to_project_key():
    from wildedge.client import WildEdge

    with (
        patch(
            "wildedge.client.default_pending_queue_dir", return_value="pending-dir"
        ) as p,
        patch("wildedge.client.default_dead_letter_dir", return_value="dead-dir") as d,
        patch(
            "wildedge.client.default_model_registry_path", return_value="registry-path"
        ) as r,
    ):
        WildEdge(dsn="https://test@test.com/proj-key")
    p.assert_called_once_with("proj-key")
    d.assert_called_once_with("proj-key")
    r.assert_called_once_with("proj-key")


def test_app_identity_override_used_for_paths():
    from wildedge.client import WildEdge

    with (
        patch(
            "wildedge.client.default_pending_queue_dir", return_value="pending-dir"
        ) as p,
        patch("wildedge.client.default_dead_letter_dir", return_value="dead-dir") as d,
        patch(
            "wildedge.client.default_model_registry_path", return_value="registry-path"
        ) as r,
    ):
        WildEdge(dsn="https://test@test.com/proj-key", app_identity="app-a")
    p.assert_called_once_with("app-a")
    d.assert_called_once_with("app-a")
    r.assert_called_once_with("app-a")


def test_app_identity_env_override_used_for_paths(monkeypatch):
    from wildedge.client import WildEdge

    monkeypatch.setenv(constants.ENV_APP_IDENTITY, "env-app")
    with (
        patch(
            "wildedge.client.default_pending_queue_dir", return_value="pending-dir"
        ) as p,
        patch("wildedge.client.default_dead_letter_dir", return_value="dead-dir") as d,
        patch(
            "wildedge.client.default_model_registry_path", return_value="registry-path"
        ) as r,
    ):
        WildEdge(dsn="https://test@test.com/proj-key")
    p.assert_called_once_with("env-app")
    d.assert_called_once_with("env-app")
    r.assert_called_once_with("env-app")
