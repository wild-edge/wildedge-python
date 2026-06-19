"""Opt-in raw input/output attachment upload.

Capture and upload are decoupled, mirroring the Swift SDK:

  track_inference(attachments=[...])        AttachmentUploader (own thread)
    -> generate attachment_id                 -> POST /api/attachments/presign
    -> AttachmentStore.append(bytes + meta)   -> PUT bytes to presigned URL
    -> event["attachments"] = [refs]          -> delete local bytes on success

The batch flush never waits on bytes: the reference is written into the
inference event immediately, while the bytes ride this module's independent
loop. Only collect attachments for events guaranteed to be transmitted - with
reservoir sampling inactive every inference event is transmitted, so every
attachment is eligible.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from wildedge import constants
from wildedge.logging import logger

AttachmentStorageStrategy = str  # "file" | "inline"


@dataclass
class Attachment:
    """A raw input or output blob to upload alongside an inference event.

    Provide exactly one of ``data`` (in-memory bytes) or ``path`` (a file on
    disk). ``content_type`` is the MIME type baked into the presigned URL and
    echoed in the inference event reference.
    """

    content_type: str
    role: str = "input"  # "input" | "output"
    data: bytes | None = None
    path: str | None = None
    name: str | None = None

    def resolve_bytes(self) -> bytes:
        if self.data is not None:
            return self.data
        if self.path is not None:
            return Path(self.path).expanduser().read_bytes()
        raise ValueError("Attachment requires either data or path")

    def size_bytes(self) -> int:
        if self.data is not None:
            return len(self.data)
        if self.path is not None:
            return Path(self.path).expanduser().stat().st_size
        raise ValueError("Attachment requires either data or path")


@dataclass
class PendingAttachment:
    """A buffered attachment awaiting upload, reconstructed from disk."""

    attachment_id: str
    inference_id: str
    role: str
    content_type: str
    size_bytes: int
    inference_timestamp: str
    registered_at: float
    record_path: Path
    bin_path: Path | None = None
    inline_data: bytes | None = None
    name: str | None = None

    def read_bytes(self) -> bytes:
        if self.inline_data is not None:
            return self.inline_data
        if self.bin_path is not None:
            return self.bin_path.read_bytes()
        raise ValueError("PendingAttachment has no payload")


class UploadOutcome(Enum):
    UPLOADED = "uploaded"
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    FEATURE_DISABLED = "feature_disabled"


class AttachmentStore:
    """Disk-backed, restart-safe buffer of pending attachments.

    Each attachment is one JSON record file (timestamp-ordered name for
    oldest-first draining); ``file`` strategy also writes a sibling ``.bin``
    holding the raw bytes, while ``inline`` embeds base64 bytes in the record.
    """

    def __init__(
        self,
        directory: str,
        *,
        strategy: AttachmentStorageStrategy = constants.DEFAULT_ATTACHMENT_STORAGE_STRATEGY,
        max_pending: int = constants.DEFAULT_MAX_PENDING_ATTACHMENTS,
    ) -> None:
        self.directory = Path(directory).expanduser()
        self.strategy = strategy
        self.max_pending = max_pending
        self.lock = threading.Lock()
        self._seq = 0
        self.directory.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        attachment: Attachment,
        attachment_id: str,
        inference_id: str,
        inference_timestamp: str,
        data: bytes,
    ) -> None:
        with self.lock:
            self._evict_for_capacity()
            record: dict[str, Any] = {
                "attachment_id": attachment_id,
                "inference_id": inference_id,
                "role": attachment.role,
                "content_type": attachment.content_type,
                "size_bytes": len(data),
                "inference_timestamp": inference_timestamp,
                "registered_at": time.time(),
                "name": attachment.name,
            }
            if self.strategy == "inline":
                record["payload_kind"] = "inline"
                record["data_b64"] = base64.b64encode(data).decode("ascii")
            else:
                bin_path = self.directory / f"{attachment_id}.bin"
                bin_path.write_bytes(data)
                record["payload_kind"] = "file"
                record["bin_file"] = bin_path.name

            name = f"{time.time_ns():020d}-{self._seq:010d}-{attachment_id}.json"
            self._seq += 1
            (self.directory / name).write_text(
                json.dumps(record, separators=(",", ":"), ensure_ascii=True)
            )

    def _evict_for_capacity(self) -> None:
        # Caller holds the lock.
        records = sorted(self.directory.glob("*.json"))
        overflow = len(records) - self.max_pending + 1
        if overflow <= 0:
            return
        for path in records[:overflow]:
            pending = self._load(path)
            if pending is not None:
                self._delete(pending)
            else:
                path.unlink(missing_ok=True)
        logger.warning(
            "wildedge: attachment buffer full (%d); dropped %d oldest",
            self.max_pending,
            overflow,
        )

    def peek_many(self, n: int) -> list[PendingAttachment]:
        with self.lock:
            result: list[PendingAttachment] = []
            for path in sorted(self.directory.glob("*.json"))[:n]:
                pending = self._load(path)
                if pending is not None:
                    result.append(pending)
                else:
                    path.unlink(missing_ok=True)
            return result

    def remove(self, pendings: list[PendingAttachment]) -> None:
        with self.lock:
            for pending in pendings:
                self._delete(pending)

    def _delete(self, pending: PendingAttachment) -> None:
        # Caller holds the lock.
        pending.record_path.unlink(missing_ok=True)
        if pending.bin_path is not None:
            pending.bin_path.unlink(missing_ok=True)

    def clear(self) -> None:
        with self.lock:
            for path in self.directory.glob("*.json"):
                pending = self._load(path)
                if pending is not None:
                    self._delete(pending)
                else:
                    path.unlink(missing_ok=True)
            for stray in self.directory.glob("*.bin"):
                stray.unlink(missing_ok=True)

    def length(self) -> int:
        with self.lock:
            return sum(1 for _ in self.directory.glob("*.json"))

    def _load(self, path: Path) -> PendingAttachment | None:
        try:
            record = json.loads(path.read_text())
        except Exception:
            return None
        if not isinstance(record, dict):
            return None
        try:
            kind = record["payload_kind"]
            bin_path: Path | None = None
            inline_data: bytes | None = None
            if kind == "file":
                bin_path = self.directory / record["bin_file"]
                if not bin_path.exists():
                    return None
            elif kind == "inline":
                inline_data = base64.b64decode(record["data_b64"])
            else:
                return None
            return PendingAttachment(
                attachment_id=str(record["attachment_id"]),
                inference_id=str(record["inference_id"]),
                role=str(record["role"]),
                content_type=str(record["content_type"]),
                size_bytes=int(record["size_bytes"]),
                inference_timestamp=str(record["inference_timestamp"]),
                registered_at=float(record["registered_at"]),
                record_path=path,
                bin_path=bin_path,
                inline_data=inline_data,
                name=record.get("name"),
            )
        except (KeyError, ValueError, TypeError):
            return None


class AttachmentManager:
    """Capture-side gate: applies the filter hook and per-inference caps, then
    buffers eligible attachments and returns the event references."""

    def __init__(
        self,
        store: AttachmentStore,
        *,
        max_per_inference: int = constants.DEFAULT_MAX_ATTACHMENTS_PER_INFERENCE,
        max_size_bytes: int = constants.DEFAULT_MAX_ATTACHMENT_SIZE_BYTES,
        attachment_filter: Callable[[list[Attachment]], list[Attachment]] | None = None,
        debug: bool = False,
    ) -> None:
        self.store = store
        self.max_per_inference = max_per_inference
        self.max_size_bytes = max_size_bytes
        self.attachment_filter = attachment_filter
        self.debug = debug
        self.enabled = True

    def disable(self) -> None:
        """Stop accepting attachments (e.g. the project has the feature off)."""
        self.enabled = False

    def capture(
        self,
        attachments: list[Attachment],
        inference_id: str,
        inference_timestamp: datetime,
    ) -> list[dict[str, str]]:
        if not self.enabled or not attachments:
            return []

        selected = attachments
        if self.attachment_filter is not None:
            try:
                selected = self.attachment_filter(list(attachments))
            except Exception as exc:  # pragma: no cover - user callback failures
                logger.warning("wildedge: attachment_filter raised: %s", exc)
                return []

        if len(selected) > self.max_per_inference:
            if self.debug:
                logger.debug(
                    "wildedge: %d attachments exceeds max_per_inference=%d; dropping excess",
                    len(selected),
                    self.max_per_inference,
                )
            selected = selected[: self.max_per_inference]

        ts = inference_timestamp.isoformat()
        refs: list[dict[str, str]] = []
        for attachment in selected:
            try:
                data = attachment.resolve_bytes()
            except Exception as exc:
                logger.warning("wildedge: could not read attachment bytes: %s", exc)
                continue
            if len(data) > self.max_size_bytes:
                if self.debug:
                    logger.debug(
                        "wildedge: attachment (%d bytes) exceeds max_size_bytes=%d; dropping",
                        len(data),
                        self.max_size_bytes,
                    )
                continue
            attachment_id = str(uuid.uuid4())
            try:
                self.store.append(
                    attachment=attachment,
                    attachment_id=attachment_id,
                    inference_id=inference_id,
                    inference_timestamp=ts,
                    data=data,
                )
            except Exception as exc:
                logger.warning("wildedge: failed to buffer attachment: %s", exc)
                continue
            refs.append(
                {
                    "attachment_id": attachment_id,
                    "role": attachment.role,
                    "content_type": attachment.content_type,
                }
            )
        return refs


class AttachmentTransmitter:
    """Presigns and PUTs a single attachment. Presign hits the ingest host with
    the project secret; the PUT goes straight to the returned blob-store URL
    with no secret."""

    def __init__(
        self,
        api_key: str,
        host: str,
        timeout: float = constants.DEFAULT_ATTACHMENT_HTTP_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.host = host.rstrip("/")
        self.timeout = timeout

    def transmit(self, pending: PendingAttachment) -> UploadOutcome:
        outcome, upload_url = self._presign(pending)
        if upload_url is None:
            return outcome
        try:
            data = pending.read_bytes()
        except Exception as exc:
            logger.warning("wildedge: attachment bytes unreadable: %s", exc)
            return UploadOutcome.PERMANENT
        return self._put(upload_url, data, pending.content_type)

    def _presign(self, pending: PendingAttachment) -> tuple[UploadOutcome, str | None]:
        url = f"{self.host}/api/attachments/presign"
        body = json.dumps(
            [
                {
                    "attachment_id": pending.attachment_id,
                    "role": pending.role,
                    "content_type": pending.content_type,
                    "size_bytes": pending.size_bytes,
                    "inference_timestamp": pending.inference_timestamp,
                }
            ]
        ).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "User-Agent": constants.SDK_VERSION,
                "X-Project-Secret": self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            raw = exc.read()
        except (urllib.error.URLError, OSError) as exc:
            logger.debug("wildedge: presign network error: %s", exc)
            return UploadOutcome.TRANSIENT, None

        if status in (200, 201):
            try:
                entries = json.loads(raw)
                upload_url = entries[0]["upload_url"]
            except (ValueError, KeyError, IndexError, TypeError):
                return UploadOutcome.TRANSIENT, None
            return UploadOutcome.UPLOADED, upload_url
        if status == 403:
            logger.warning(
                "wildedge: attachments not enabled for this project (403); "
                "disabling attachment upload"
            )
            return UploadOutcome.FEATURE_DISABLED, None
        if status == 422:
            logger.warning(
                "wildedge: attachment presign validation error (422): %s",
                raw[: constants.ERROR_MSG_MAX_LEN],
            )
            return UploadOutcome.PERMANENT, None
        if status == 401 or status == 400:
            logger.warning(
                "wildedge: attachment presign rejected (%d) - dropping", status
            )
            return UploadOutcome.PERMANENT, None
        # 429 / 5xx / anything else: retry later.
        return UploadOutcome.TRANSIENT, None

    def _put(self, upload_url: str, data: bytes, content_type: str) -> UploadOutcome:
        req = urllib.request.Request(
            upload_url,
            data=data,
            headers={"Content-Type": content_type},
            method="PUT",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        except (urllib.error.URLError, OSError) as exc:
            logger.debug("wildedge: attachment upload network error: %s", exc)
            return UploadOutcome.TRANSIENT

        if status in (200, 201, 202, 204):
            return UploadOutcome.UPLOADED
        if status == 400:
            return UploadOutcome.PERMANENT
        # Expired/invalid signature (403) and 5xx: retry, which re-presigns.
        return UploadOutcome.TRANSIENT


class AttachmentUploader:
    """Background daemon thread that drains the AttachmentStore oldest-first."""

    def __init__(
        self,
        store: AttachmentStore,
        transmitter: AttachmentTransmitter,
        *,
        flush_interval_s: float = constants.DEFAULT_ATTACHMENT_FLUSH_INTERVAL_SEC,
        max_age_s: float = constants.DEFAULT_MAX_ATTACHMENT_AGE_SEC,
        on_feature_disabled: Callable[[], None] | None = None,
        debug: bool = False,
    ) -> None:
        self.store = store
        self.transmitter = transmitter
        self.flush_interval_s = flush_interval_s
        self.max_age_s = max_age_s
        self.on_feature_disabled = on_feature_disabled
        self.debug = debug
        self.uploaded_count = 0
        self.dropped_count = 0
        self.stop_event = threading.Event()
        self.thread = self._make_thread()

    def _make_thread(self) -> threading.Thread:
        return threading.Thread(
            target=self._run, daemon=True, name="wildedge-attachment-uploader"
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _pause(self) -> None:
        """Stop the thread before a fork() so no wildedge threads survive."""
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _resume(self) -> None:
        """Start a fresh thread after a fork() in parent and child."""
        self.stop_event = threading.Event()
        self.thread = self._make_thread()
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(self.flush_interval_s):
            try:
                self.process_pending()
            except Exception as exc:  # pragma: no cover - never kill the thread
                logger.warning("wildedge: attachment uploader error: %s", exc)

    def process_pending(self) -> None:
        # Evict stale buffered bytes from the front of the queue.
        cutoff = time.time() - self.max_age_s
        batch = self.store.peek_many(constants.ATTACHMENT_PRESIGN_BATCH_SIZE)
        stale = [p for p in batch if p.registered_at < cutoff]
        if stale:
            self.store.remove(stale)
            self.dropped_count += len(stale)
            logger.warning(
                "wildedge: evicted %d stale attachment(s) (age > %.0fs)",
                len(stale),
                self.max_age_s,
            )
            batch = [p for p in batch if p.registered_at >= cutoff]

        done: list[PendingAttachment] = []
        for pending in batch:
            outcome = self.transmitter.transmit(pending)
            if outcome is UploadOutcome.UPLOADED:
                done.append(pending)
                self.uploaded_count += 1
            elif outcome is UploadOutcome.PERMANENT:
                done.append(pending)
                self.dropped_count += 1
            elif outcome is UploadOutcome.FEATURE_DISABLED:
                if self.on_feature_disabled is not None:
                    self.on_feature_disabled()
                self.store.remove(done)
                self.store.clear()
                self.stop_event.set()
                return
            else:  # TRANSIENT - stop, preserve order, retry next tick
                break

        if done:
            self.store.remove(done)
