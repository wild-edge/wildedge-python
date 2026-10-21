from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from wildedge import constants, modalities
from wildedge.attachments import Attachment, AttachmentManager, AttachmentStore
from wildedge.client import WildEdge
from wildedge.model import ModelHandle, ModelInfo


@pytest.mark.parametrize(
    ("content_type", "modality"),
    [
        ("text/plain", "text"),
        ("text/plain; charset=utf-8", "text"),
        ("application/json", "text"),
        ("application/ld+json", "text"),
        ("image/jpeg", "image"),
        ("audio/wav", "audio"),
        ("AUDIO/MPEG", "audio"),
        ("video/mp4", "video"),
        ("application/octet-stream", "other"),
        ("application/x-npy", "other"),
    ],
)
def test_modality_follows_content_type(content_type, modality):
    assert modalities.from_content_type(content_type) == modality
    assert Attachment(content_type=content_type, data=b"x").modality == modality


def test_explicit_modality_overrides_content_type():
    attachment = Attachment(
        content_type="application/octet-stream", data=b"x", modality="audio"
    )
    assert attachment.modality == "audio"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("text,audio", {"text", "audio"}),
        (" Text , AUDIO ", {"text", "audio"}),
        ("all", set(modalities.ALL)),
        (["image"], {"image"}),
        ((), set()),
    ],
)
def test_parse_accepts_lists_and_all(value, expected):
    assert modalities.parse(value) == expected


def test_parse_rejects_unknown_names():
    with pytest.raises(ValueError, match="audios"):
        modalities.parse("text,audios")


def capture(manager, *attachments):
    return manager.capture(list(attachments), "inf-1", datetime.now(timezone.utc))


def text():
    return Attachment(content_type="text/plain", data=b"hi")


def image():
    return Attachment(content_type="image/jpeg", data=b"img")


def test_default_uploads_text_only_and_warns_once(tmp_path, caplog):
    store = AttachmentStore(str(tmp_path))
    manager = AttachmentManager(store)
    with caplog.at_level(logging.DEBUG, logger="wildedge"):
        refs = capture(manager, image(), text())
        capture(manager, image())
    assert [ref["content_type"] for ref in refs] == ["text/plain"]
    assert store.length() == 1
    warnings = [r for r in caplog.records if "image attachments" in r.message]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING


def test_explicit_choice_logs_blocked_modalities_at_debug(tmp_path, caplog):
    manager = AttachmentManager(
        AttachmentStore(str(tmp_path)), allowed_modalities={"text"}
    )
    with caplog.at_level(logging.DEBUG, logger="wildedge"):
        capture(manager, image())
    (record,) = [r for r in caplog.records if "image attachments" in r.message]
    assert record.levelno == logging.DEBUG


def test_filter_never_sees_blocked_attachments(tmp_path):
    seen = []
    manager = AttachmentManager(
        AttachmentStore(str(tmp_path)),
        attachment_filter=lambda items: seen.extend(items) or items,
    )
    capture(manager, image(), text())
    assert [a.modality for a in seen] == ["text"]


def test_allowed_modalities_upload(tmp_path):
    manager = AttachmentManager(
        AttachmentStore(str(tmp_path)), allowed_modalities={"image"}
    )
    refs = capture(manager, image(), text())
    assert [ref["content_type"] for ref in refs] == ["image/jpeg"]


def make_handle(**kwargs):
    info = ModelInfo(
        model_name="m", model_version="1", model_source="local", model_format="onnx"
    )
    return ModelHandle("m", info, lambda _event: None, **kwargs)


def test_handle_wants_nothing_without_attachments():
    assert not make_handle().wants_attachment("text")


def test_handle_asks_the_client_policy(tmp_path):
    manager = AttachmentManager(AttachmentStore(str(tmp_path)))
    handle = make_handle(
        capture_attachments=manager.capture, accepts_attachment=manager.accepts
    )
    assert handle.wants_attachment("text")
    assert not handle.wants_attachment("audio")


def client(tmp_path, **kwargs):
    return WildEdge(
        dsn="https://test@test.com/key",
        attachments_enabled=True,
        attachment_dir=str(tmp_path),
        sampling_interval_s=0,
        **kwargs,
    )


def test_client_takes_modalities_argument(tmp_path):
    wildedge = client(tmp_path, attachment_modalities=["text", "audio"])
    try:
        assert wildedge.attachment_manager.allowed_modalities == {"text", "audio"}
    finally:
        wildedge.close()


def test_client_reads_modalities_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv(constants.ENV_ATTACHMENT_MODALITIES, "all")
    wildedge = client(tmp_path)
    try:
        assert wildedge.attachment_manager.allowed_modalities == set(modalities.ALL)
    finally:
        wildedge.close()


def test_client_defaults_to_text(tmp_path, monkeypatch):
    monkeypatch.delenv(constants.ENV_ATTACHMENT_MODALITIES, raising=False)
    wildedge = client(tmp_path)
    try:
        assert wildedge.attachment_manager.allowed_modalities == {"text"}
    finally:
        wildedge.close()


def test_client_rejects_unknown_modality(tmp_path):
    with pytest.raises(ValueError, match="pictures"):
        client(tmp_path, attachment_modalities="pictures")
