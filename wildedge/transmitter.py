"""HTTP transmitter: sends batches to /api/ingest (stdlib-only)."""

from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from wildedge import config
from wildedge.logging import logger


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Raise immediately on any redirect instead of following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: PLR0913
        raise urllib.error.HTTPError(newurl, code, msg, headers, fp)


@dataclass
class IngestResponse:
    status: str
    batch_id: str
    events_accepted: int
    events_rejected: int
    server_time: str | None = None
    rejected: list[dict] | None = None


class TransmitError(Exception):
    """Raised for retryable errors (429 / 5xx / network)."""


class Transmitter:
    """Sends batch envelopes to the WildEdge ingest endpoint (stdlib urllib)."""

    def __init__(
        self, api_key: str, host: str, timeout: float = config.DEFAULT_HTTP_TIMEOUT
    ):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "User-Agent": config.SDK_VERSION,
            "X-Project-Secret": api_key,
            "Content-Type": "application/json",
        }
        self._opener = urllib.request.build_opener(_NoRedirectHandler)

    def send(self, batch: dict) -> IngestResponse:
        url = f"{self.host}/api/ingest"
        body = gzip.compress(json.dumps(batch).encode())
        req = urllib.request.Request(
            url,
            data=body,
            headers={**self.headers, "Content-Encoding": "gzip"},
            method="POST",
        )

        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                status_code = resp.status
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            raw = exc.read()
        except (urllib.error.URLError, OSError) as exc:
            raise TransmitError(f"Network error: {exc}") from exc

        if status_code == 202:
            data = json.loads(raw)
            return IngestResponse(
                status=data.get("status", "accepted"),
                batch_id=data.get("batch_id", ""),
                events_accepted=data.get("events_accepted", 0),
                events_rejected=data.get("events_rejected", 0),
                server_time=data.get("server_time"),
                rejected=data.get("rejected"),
            )

        if status_code == 400:
            logger.warning(
                "wildedge: batch rejected (400) - discarding: %s",
                raw[: config.ERROR_MSG_MAX_LEN],
            )
            return IngestResponse(
                status="rejected",
                batch_id=batch.get("batch_id", ""),
                events_accepted=0,
                events_rejected=len(batch.get("events", [])),
            )

        if status_code == 401:
            logger.error("wildedge: authentication failed (401) - check your API key")
            return IngestResponse(
                status="unauthorized",
                batch_id=batch.get("batch_id", ""),
                events_accepted=0,
                events_rejected=len(batch.get("events", [])),
            )

        if 300 <= status_code < 400:
            # Redirects should never occur (we disable redirect following).
            # Treat as a permanent config error so we don't loop.
            logger.error(
                "wildedge: unexpected redirect (%d) to %s; check WILDEDGE_DSN",
                status_code,
                raw[: config.ERROR_MSG_MAX_LEN],
            )
            return IngestResponse(
                status="error",
                batch_id=batch.get("batch_id", ""),
                events_accepted=0,
                events_rejected=len(batch.get("events", [])),
            )

        if status_code == 404:
            logger.error(
                "wildedge: endpoint not found (404) at %s; check WILDEDGE_DSN", url
            )
            return IngestResponse(
                status="error",
                batch_id=batch.get("batch_id", ""),
                events_accepted=0,
                events_rejected=len(batch.get("events", [])),
            )

        if status_code == 429 or status_code >= 500:
            raise TransmitError(
                f"HTTP {status_code}: {raw[: config.ERROR_MSG_MAX_LEN]!r}"
            )

        if 400 <= status_code < 500:
            # Other 4xx (e.g. 422 Unprocessable) are permanent client errors; discard.
            logger.warning(
                "wildedge: batch rejected (%d) - discarding: %s",
                status_code,
                raw[: config.ERROR_MSG_MAX_LEN],
            )
            return IngestResponse(
                status="rejected",
                batch_id=batch.get("batch_id", ""),
                events_accepted=0,
                events_rejected=len(batch.get("events", [])),
            )

    def close(self) -> None:
        pass
