from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from wildedge.attachments import (
    Attachment,
    AttachmentManager,
    AttachmentStore,
    AttachmentTransmitter,
    AttachmentUploader,
    PendingAttachment,
    UploadOutcome,
)
from wildedge.events.inference import InferenceEvent
from wildedge.model import ModelHandle, ModelInfo

# --------------------------------------------------------------------------- #
# AttachmentStore
# --------------------------------------------------------------------------- #


def _append(store: AttachmentStore, data: bytes, attachment_id: str, role="input"):
    store.append(
        attachment=Attachment(content_type="text/plain", role=role, data=data),
        attachment_id=attachment_id,
        inference_id="inf-1",
        inference_timestamp="2026-01-17T10:30:00+00:00",
        data=data,
    )


def test_store_file_strategy_roundtrip(tmp_path):
    store = AttachmentStore(str(tmp_path), strategy="file")
    _append(store, b"hello", "a1")
    assert store.length() == 1
    pending = store.peek_many(10)
    assert len(pending) == 1
    assert pending[0].attachment_id == "a1"
    assert pending[0].read_bytes() == b"hello"
    assert pending[0].bin_path is not None and pending[0].bin_path.exists()


def test_store_inline_strategy_roundtrip(tmp_path):
    store = AttachmentStore(str(tmp_path), strategy="inline")
    _append(store, b"\x00\x01\x02inline", "a1")
    pending = store.peek_many(10)
    assert pending[0].read_bytes() == b"\x00\x01\x02inline"
    assert pending[0].bin_path is None


def test_store_survives_restart(tmp_path):
    store = AttachmentStore(str(tmp_path), strategy="file")
    _append(store, b"persisted", "a1")
    # Fresh instance reads the same directory.
    reopened = AttachmentStore(str(tmp_path), strategy="file")
    pending = reopened.peek_many(10)
    assert len(pending) == 1
    assert pending[0].read_bytes() == b"persisted"


def test_store_capacity_drops_oldest(tmp_path):
    store = AttachmentStore(str(tmp_path), strategy="file", max_pending=2)
    _append(store, b"1", "a1")
    _append(store, b"2", "a2")
    _append(store, b"3", "a3")
    assert store.length() == 2
    ids = [p.attachment_id for p in store.peek_many(10)]
    assert ids == ["a2", "a3"]
    # Oldest bin file is gone.
    assert not (tmp_path / "a1.bin").exists()


def test_store_remove_deletes_record_and_bin(tmp_path):
    store = AttachmentStore(str(tmp_path), strategy="file")
    _append(store, b"x", "a1")
    pending = store.peek_many(10)
    store.remove(pending)
    assert store.length() == 0
    assert not (tmp_path / "a1.bin").exists()


def test_store_peek_oldest_first(tmp_path):
    store = AttachmentStore(str(tmp_path), strategy="file")
    for i in range(5):
        _append(store, str(i).encode(), f"a{i}")
    ids = [p.attachment_id for p in store.peek_many(3)]
    assert ids == ["a0", "a1", "a2"]


# --------------------------------------------------------------------------- #
# AttachmentManager (capture)
# --------------------------------------------------------------------------- #


def _manager(tmp_path, **kwargs):
    store = AttachmentStore(str(tmp_path), strategy="file")
    return store, AttachmentManager(store, **kwargs)


def test_manager_capture_returns_refs_and_buffers(tmp_path):
    store, mgr = _manager(tmp_path)
    refs = mgr.capture(
        [
            Attachment(content_type="image/jpeg", role="input", data=b"img"),
            Attachment(content_type="text/plain", role="output", data=b"out"),
        ],
        "inf-1",
        datetime.now(timezone.utc),
    )
    assert len(refs) == 2
    assert refs[0] == {
        "attachment_id": refs[0]["attachment_id"],
        "role": "input",
        "content_type": "image/jpeg",
    }
    assert store.length() == 2


def test_manager_disabled_returns_nothing(tmp_path):
    store, mgr = _manager(tmp_path)
    mgr.disable()
    refs = mgr.capture(
        [Attachment(content_type="text/plain", data=b"x")],
        "inf-1",
        datetime.now(timezone.utc),
    )
    assert refs == []
    assert store.length() == 0


def test_manager_enforces_max_per_inference(tmp_path):
    store, mgr = _manager(tmp_path, max_per_inference=2)
    refs = mgr.capture(
        [Attachment(content_type="text/plain", data=b"x") for _ in range(5)],
        "inf-1",
        datetime.now(timezone.utc),
    )
    assert len(refs) == 2
    assert store.length() == 2


def test_manager_drops_oversized(tmp_path):
    store, mgr = _manager(tmp_path, max_size_bytes=4)
    refs = mgr.capture(
        [
            Attachment(content_type="text/plain", data=b"ok"),
            Attachment(content_type="text/plain", data=b"way too big"),
        ],
        "inf-1",
        datetime.now(timezone.utc),
    )
    assert len(refs) == 1
    assert store.length() == 1


def test_manager_filter_hook(tmp_path):
    store, mgr = _manager(
        tmp_path,
        attachment_filter=lambda atts: [a for a in atts if a.role == "input"],
    )
    refs = mgr.capture(
        [
            Attachment(content_type="text/plain", role="input", data=b"keep"),
            Attachment(content_type="text/plain", role="output", data=b"drop"),
        ],
        "inf-1",
        datetime.now(timezone.utc),
    )
    assert len(refs) == 1
    assert refs[0]["role"] == "input"


# --------------------------------------------------------------------------- #
# InferenceEvent attachment refs
# --------------------------------------------------------------------------- #


def test_inference_event_emits_attachments():
    event = InferenceEvent(
        model_id="m1",
        duration_ms=10,
        attachments=[
            {"attachment_id": "a1", "role": "input", "content_type": "image/jpeg"}
        ],
    )
    data = event.to_dict()
    assert data["attachments"] == [
        {"attachment_id": "a1", "role": "input", "content_type": "image/jpeg"}
    ]


def test_inference_event_omits_attachments_when_absent():
    event = InferenceEvent(model_id="m1", duration_ms=10)
    assert "attachments" not in event.to_dict()


def test_text_input_meta_no_has_attachments():
    from wildedge.events.inference import TextInputMeta

    assert not hasattr(TextInputMeta(char_count=1), "has_attachments")


# --------------------------------------------------------------------------- #
# ModelHandle wiring
# --------------------------------------------------------------------------- #


def _make_handle(tmp_path):
    store = AttachmentStore(str(tmp_path), strategy="file")
    mgr = AttachmentManager(store)
    published: list[dict] = []
    info = ModelInfo(
        model_name="m", model_version="1", model_source="local", model_format="onnx"
    )
    handle = ModelHandle("m1", info, published.append, capture_attachments=mgr.capture)
    return handle, store, published


def test_track_inference_wires_attachments(tmp_path):
    handle, store, published = _make_handle(tmp_path)
    handle.track_inference(
        duration_ms=10,
        input_modality="image",
        attachments=[Attachment(content_type="image/jpeg", role="input", data=b"img")],
    )
    assert store.length() == 1
    event = published[0]
    assert len(event["attachments"]) == 1
    assert event["attachments"][0]["content_type"] == "image/jpeg"
    # The buffered attachment is keyed to this inference event.
    assert store.peek_many(1)[0].inference_id == event["inference"]["inference_id"]


def test_track_inference_without_capture_callback_omits_refs(tmp_path):
    info = ModelInfo(
        model_name="m", model_version="1", model_source="local", model_format="onnx"
    )
    published: list[dict] = []
    handle = ModelHandle("m1", info, published.append)  # no capture callback
    handle.track_inference(
        duration_ms=10,
        attachments=[Attachment(content_type="image/jpeg", data=b"img")],
    )
    assert "attachments" not in published[0]


# --------------------------------------------------------------------------- #
# AttachmentTransmitter (presign + PUT) with a fake urlopen
# --------------------------------------------------------------------------- #


class _Resp:
    def __init__(self, status: int, body: bytes = b""):
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _fake_urlopen(responses):
    """responses: list of _Resp or Exception, consumed in call order."""
    calls = iter(responses)

    def opener(req, timeout=None):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    return opener


PRESIGN_OK = _Resp(
    200,
    json.dumps([{"upload_url": "https://blob/put", "attachment_id": "a1"}]).encode(),
)


def _pending(tmp_path) -> PendingAttachment:
    store = AttachmentStore(str(tmp_path), strategy="file")
    _append(store, b"payload", "a1")
    return store.peek_many(1)[0]


def _transmit(monkeypatch, tmp_path, responses) -> UploadOutcome:
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(responses))
    tx = AttachmentTransmitter(api_key="secret", host="https://ingest.example")
    return tx.transmit(_pending(tmp_path))


def test_transmit_success(monkeypatch, tmp_path):
    outcome = _transmit(monkeypatch, tmp_path, [PRESIGN_OK, _Resp(200)])
    assert outcome is UploadOutcome.UPLOADED


def test_transmit_put_204(monkeypatch, tmp_path):
    outcome = _transmit(monkeypatch, tmp_path, [PRESIGN_OK, _Resp(204)])
    assert outcome is UploadOutcome.UPLOADED


def test_presign_403_feature_disabled(monkeypatch, tmp_path):
    err = urllib.error.HTTPError("u", 403, "forbidden", {}, None)
    outcome = _transmit(monkeypatch, tmp_path, [err])
    assert outcome is UploadOutcome.FEATURE_DISABLED


def test_presign_422_permanent(monkeypatch, tmp_path):
    err = urllib.error.HTTPError("u", 422, "bad", {}, None)
    outcome = _transmit(monkeypatch, tmp_path, [err])
    assert outcome is UploadOutcome.PERMANENT


def test_presign_429_transient(monkeypatch, tmp_path):
    err = urllib.error.HTTPError("u", 429, "rate", {}, None)
    outcome = _transmit(monkeypatch, tmp_path, [err])
    assert outcome is UploadOutcome.TRANSIENT


def test_presign_500_transient(monkeypatch, tmp_path):
    err = urllib.error.HTTPError("u", 500, "boom", {}, None)
    outcome = _transmit(monkeypatch, tmp_path, [err])
    assert outcome is UploadOutcome.TRANSIENT


def test_presign_network_error_transient(monkeypatch, tmp_path):
    outcome = _transmit(monkeypatch, tmp_path, [urllib.error.URLError("offline")])
    assert outcome is UploadOutcome.TRANSIENT


def test_put_expired_signature_transient(monkeypatch, tmp_path):
    # Blob store returns 403 on an expired URL -> retry (re-presign next tick).
    err = urllib.error.HTTPError("u", 403, "expired", {}, None)
    outcome = _transmit(monkeypatch, tmp_path, [PRESIGN_OK, err])
    assert outcome is UploadOutcome.TRANSIENT


def test_put_400_permanent(monkeypatch, tmp_path):
    err = urllib.error.HTTPError("u", 400, "bad", {}, None)
    outcome = _transmit(monkeypatch, tmp_path, [PRESIGN_OK, err])
    assert outcome is UploadOutcome.PERMANENT


# --------------------------------------------------------------------------- #
# AttachmentUploader
# --------------------------------------------------------------------------- #


class _StubTransmitter:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.seen: list[str] = []

    def transmit(self, pending):
        self.seen.append(pending.attachment_id)
        return self.outcomes.pop(0)


def _filled_store(tmp_path, n=3):
    store = AttachmentStore(str(tmp_path), strategy="file")
    for i in range(n):
        _append(store, str(i).encode(), f"a{i}")
    return store


def test_uploader_uploads_and_removes(tmp_path):
    store = _filled_store(tmp_path, 3)
    tx = _StubTransmitter([UploadOutcome.UPLOADED] * 3)
    up = AttachmentUploader(store, tx)
    up.process_pending()
    assert store.length() == 0
    assert up.uploaded_count == 3


def test_uploader_stops_at_transient(tmp_path):
    store = _filled_store(tmp_path, 3)
    tx = _StubTransmitter(
        [UploadOutcome.UPLOADED, UploadOutcome.TRANSIENT, UploadOutcome.UPLOADED]
    )
    up = AttachmentUploader(store, tx)
    up.process_pending()
    # First removed, second hit transient -> stop; a1 and a2 remain.
    ids = [p.attachment_id for p in store.peek_many(10)]
    assert ids == ["a1", "a2"]
    assert up.uploaded_count == 1


def test_uploader_drops_on_permanent(tmp_path):
    store = _filled_store(tmp_path, 2)
    tx = _StubTransmitter([UploadOutcome.PERMANENT, UploadOutcome.UPLOADED])
    up = AttachmentUploader(store, tx)
    up.process_pending()
    assert store.length() == 0
    assert up.uploaded_count == 1
    assert up.dropped_count == 1


def test_uploader_feature_disabled_clears_and_callbacks(tmp_path):
    store = _filled_store(tmp_path, 3)
    tx = _StubTransmitter([UploadOutcome.FEATURE_DISABLED])
    disabled = []
    up = AttachmentUploader(
        store, tx, on_feature_disabled=lambda: disabled.append(True)
    )
    up.process_pending()
    assert disabled == [True]
    assert store.length() == 0


def test_uploader_evicts_stale(tmp_path):
    store = _filled_store(tmp_path, 2)
    tx = _StubTransmitter([UploadOutcome.UPLOADED] * 2)
    up = AttachmentUploader(store, tx, max_age_s=0.0)
    time.sleep(0.01)
    up.process_pending()
    assert store.length() == 0
    assert up.dropped_count == 2
    assert up.uploaded_count == 0  # all evicted before upload
    assert tx.seen == []


# --------------------------------------------------------------------------- #
# Client wiring
# --------------------------------------------------------------------------- #


def test_client_attachments_disabled_by_default(monkeypatch):
    from wildedge import constants
    from wildedge.client import WildEdge

    monkeypatch.delenv(constants.ENV_ATTACHMENTS_ENABLED, raising=False)
    client = WildEdge(dsn="https://test@test.com/key")
    try:
        assert client.attachment_manager is None
        assert client.attachment_uploader is None
    finally:
        client.close()


def test_client_attachments_enabled_wires_capture(monkeypatch, tmp_path):
    from wildedge.client import WildEdge

    client = WildEdge(
        dsn="https://test@test.com/key",
        attachments_enabled=True,
        attachment_dir=str(tmp_path),
        sampling_interval_s=0,
    )
    try:
        assert client.attachment_manager is not None
        handle = client.register_model(object(), model_id="m1", source="local")
        assert handle.capture_attachments is not None
    finally:
        client.close()


def test_client_attachments_enabled_via_env(monkeypatch, tmp_path):
    from wildedge import constants
    from wildedge.client import WildEdge

    monkeypatch.setenv(constants.ENV_ATTACHMENTS_ENABLED, "1")
    client = WildEdge(
        dsn="https://test@test.com/key",
        attachment_dir=str(tmp_path),
        sampling_interval_s=0,
    )
    try:
        assert client.attachment_manager is not None
    finally:
        client.close()
