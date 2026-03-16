"""Tests for batch envelope builder."""

from datetime import datetime, timezone

from wildedge import constants
from wildedge.batch import build_batch
from wildedge.platforms.device_info import DeviceInfo


def make_device() -> DeviceInfo:
    return DeviceInfo(
        app_version="1.0.0",
        device_id="device-123",
        device_type="linux",
    )


class TestBuildBatch:
    def test_returns_protocol_version(self):
        batch = build_batch(
            device=make_device(),
            models={},
            events=[],
            session_id="sess-1",
            created_at=datetime.now(timezone.utc),
        )
        assert batch["protocol_version"] == constants.PROTOCOL_VERSION

    def test_includes_device(self):
        batch = build_batch(
            device=make_device(),
            models={},
            events=[],
            session_id="sess-1",
            created_at=datetime.now(timezone.utc),
        )
        assert batch["device"]["device_id"] == "device-123"

    def test_includes_models(self):
        models = {"my-model": {"model_name": "test"}}
        batch = build_batch(
            device=make_device(),
            models=models,
            events=[],
            session_id="sess-1",
            created_at=datetime.now(timezone.utc),
        )
        assert batch["models"] == models

    def test_includes_events(self):
        events = [{"event_type": "inference"}]
        batch = build_batch(
            device=make_device(),
            models={},
            events=events,
            session_id="sess-1",
            created_at=datetime.now(timezone.utc),
        )
        assert batch["events"] == events

    def test_internal_queue_fields_are_not_sent(self):
        events = [
            {
                "event_type": "inference",
                "__we_first_queued_at": 1.0,
                "__we_attempts": 3,
            }
        ]
        batch = build_batch(
            device=make_device(),
            models={},
            events=events,
            session_id="sess-1",
            created_at=datetime.now(timezone.utc),
        )
        assert "__we_first_queued_at" not in batch["events"][0]
        assert "__we_attempts" not in batch["events"][0]

    def test_batch_id_is_unique(self):
        now = datetime.now(timezone.utc)
        b1 = build_batch(make_device(), {}, [], "s", now)
        b2 = build_batch(make_device(), {}, [], "s", now)
        assert b1["batch_id"] != b2["batch_id"]

    def test_sent_at_is_present(self):
        batch = build_batch(
            device=make_device(),
            models={},
            events=[],
            session_id="sess-1",
            created_at=datetime.now(timezone.utc),
        )
        assert "sent_at" in batch
        assert batch["sent_at"]

    def test_session_id_preserved(self):
        batch = build_batch(
            device=make_device(),
            models={},
            events=[],
            session_id="my-session-id",
            created_at=datetime.now(timezone.utc),
        )
        assert batch["session_id"] == "my-session-id"
