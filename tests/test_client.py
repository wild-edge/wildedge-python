"""Tests for WildEdge client configuration validation."""

from unittest.mock import patch

import pytest

from wildedge import config


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all dependencies to isolate validation tests."""
    with (
        patch("wildedge.client.detect_device"),
        patch("wildedge.client.Transmitter"),
        patch("wildedge.client.Consumer"),
        patch("wildedge.client.EventQueue"),
        patch("wildedge.client.ModelRegistry"),
    ):
        yield


def test_batch_size_too_low():
    from wildedge.client import WildEdge

    with pytest.raises(
        ValueError,
        match=f"batch_size must be between {config.BATCH_SIZE_MIN} and {config.BATCH_SIZE_MAX}",
    ):
        WildEdge(dsn="https://test@test.com/key", batch_size=0)


def test_batch_size_too_high():
    from wildedge.client import WildEdge

    with pytest.raises(
        ValueError,
        match=f"batch_size must be between {config.BATCH_SIZE_MIN} and {config.BATCH_SIZE_MAX}",
    ):
        WildEdge(dsn="https://test@test.com/key", batch_size=101)


def test_flush_interval_too_low():
    from wildedge.client import WildEdge

    with pytest.raises(
        ValueError,
        match=f"flush_interval_sec must be between {config.FLUSH_INTERVAL_MIN} and {config.FLUSH_INTERVAL_MAX}",
    ):
        WildEdge(dsn="https://test@test.com/key", flush_interval_sec=0)


def test_flush_interval_too_high():
    from wildedge.client import WildEdge

    with pytest.raises(
        ValueError,
        match=f"flush_interval_sec must be between {config.FLUSH_INTERVAL_MIN} and {config.FLUSH_INTERVAL_MAX}",
    ):
        WildEdge(dsn="https://test@test.com/key", flush_interval_sec=3601)


def test_max_queue_size_too_low():
    from wildedge.client import WildEdge

    with pytest.raises(
        ValueError,
        match=f"max_queue_size must be between {config.MAX_QUEUE_SIZE_MIN} and {config.MAX_QUEUE_SIZE_MAX}",
    ):
        WildEdge(dsn="https://test@test.com/key", max_queue_size=9)


def test_max_queue_size_too_high():
    from wildedge.client import WildEdge

    with pytest.raises(
        ValueError,
        match=f"max_queue_size must be between {config.MAX_QUEUE_SIZE_MIN} and {config.MAX_QUEUE_SIZE_MAX}",
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
