"""Tests for the HTTP transmitter (stdlib urllib)."""

import json
import urllib.error
import urllib.request
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from wildedge.transmitter import IngestResponse, TransmitError, Transmitter


def _make_response(status_code: int, body: dict | str) -> MagicMock:
    """Create a mock response object compatible with opener.open()."""
    if isinstance(body, dict):
        raw = json.dumps(body).encode()
    else:
        raw = body.encode() if isinstance(body, str) else body

    resp = MagicMock()
    resp.status = status_code
    resp.read.return_value = raw
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_http_error(status_code: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://test",
        code=status_code,
        msg="",
        hdrs=None,  # type: ignore[arg-type]
        fp=BytesIO(body.encode()),
    )


class TestTransmitter:
    def test_send_202_returns_ingest_response(self):
        body = {
            "status": "accepted",
            "batch_id": "batch-123",
            "events_accepted": 5,
            "events_rejected": 0,
            "server_time": "2026-01-01T00:00:00Z",
        }
        t = Transmitter(api_key="test-key", host="https://app.wildedge.dev")
        with patch.object(t._opener, "open", return_value=_make_response(202, body)):
            resp = t.send({"batch_id": "batch-123", "events": []})

        assert isinstance(resp, IngestResponse)
        assert resp.status == "accepted"
        assert resp.events_accepted == 5
        assert resp.batch_id == "batch-123"

    def test_send_400_returns_rejected(self):
        t = Transmitter(api_key="test-key", host="https://app.wildedge.dev")
        with patch.object(
            t._opener, "open", side_effect=_make_http_error(400, "Bad request")
        ):
            resp = t.send({"batch_id": "b-1", "events": [1, 2, 3]})

        assert resp.status == "rejected"
        assert resp.events_rejected == 3

    def test_send_401_returns_unauthorized(self):
        t = Transmitter(api_key="bad-key", host="https://app.wildedge.dev")
        with patch.object(
            t._opener, "open", side_effect=_make_http_error(401, "Unauthorized")
        ):
            resp = t.send({"batch_id": "b-1", "events": []})

        assert resp.status == "unauthorized"

    def test_send_429_raises_transmit_error(self):
        t = Transmitter(api_key="test-key", host="https://app.wildedge.dev")
        with patch.object(
            t._opener, "open", side_effect=_make_http_error(429, "Rate limited")
        ):
            with pytest.raises(TransmitError):
                t.send({"batch_id": "b-1", "events": []})

    def test_send_500_raises_transmit_error(self):
        t = Transmitter(api_key="test-key", host="https://app.wildedge.dev")
        with patch.object(
            t._opener, "open", side_effect=_make_http_error(500, "Server error")
        ):
            with pytest.raises(TransmitError):
                t.send({"batch_id": "b-1", "events": []})

    def test_network_error_raises_transmit_error(self):
        t = Transmitter(api_key="test-key", host="https://app.wildedge.dev")
        with patch.object(
            t._opener,
            "open",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(TransmitError):
                t.send({"batch_id": "b-1", "events": []})

    def test_auth_header_sent(self):
        captured = {}

        def fake_open(req, timeout=None):
            captured["headers"] = dict(req.headers)
            return _make_response(
                202,
                {
                    "status": "accepted",
                    "batch_id": "b-1",
                    "events_accepted": 0,
                    "events_rejected": 0,
                },
            )

        t = Transmitter(api_key="my-secret", host="https://app.wildedge.dev")
        with patch.object(t._opener, "open", side_effect=fake_open):
            t.send({"batch_id": "b-1", "events": []})

        # urllib capitalises headers: X-project-secret
        headers_lower = {k.lower(): v for k, v in captured["headers"].items()}
        assert headers_lower.get("x-project-secret") == "my-secret"

    def test_user_agent_header_sent(self):
        captured = {}

        def fake_open(req, timeout=None):
            captured["headers"] = dict(req.headers)
            return _make_response(
                202,
                {
                    "status": "accepted",
                    "batch_id": "b-1",
                    "events_accepted": 0,
                    "events_rejected": 0,
                },
            )

        t = Transmitter(api_key="k", host="https://app.wildedge.dev")
        with patch.object(t._opener, "open", side_effect=fake_open):
            t.send({"batch_id": "b-1", "events": []})

        headers_lower = {k.lower(): v for k, v in captured["headers"].items()}
        assert "wildedge-python" in headers_lower.get("user-agent", "")

    def test_trailing_slash_in_host_handled(self):
        captured = {}

        def fake_open(req, timeout=None):
            captured["req"] = req
            return _make_response(
                202,
                {
                    "status": "accepted",
                    "batch_id": "b-1",
                    "events_accepted": 0,
                    "events_rejected": 0,
                },
            )

        t = Transmitter(api_key="k", host="https://app.wildedge.dev/")
        with patch.object(t._opener, "open", side_effect=fake_open):
            t.send({"batch_id": "b-1", "events": []})

        assert captured["req"].full_url == "https://app.wildedge.dev/api/ingest"

    def test_send_404_returns_error_no_retry(self):
        t = Transmitter(api_key="test-key", host="https://app.wildedge.dev")
        with patch.object(
            t._opener, "open", side_effect=_make_http_error(404, "Not found")
        ):
            resp = t.send({"batch_id": "b-1", "events": []})

        assert resp.status == "error"

    def test_send_301_redirect_returns_error_no_retry(self):
        t = Transmitter(api_key="test-key", host="https://app.wildedge.dev")
        with patch.object(
            t._opener, "open", side_effect=_make_http_error(301, "Moved")
        ):
            resp = t.send({"batch_id": "b-1", "events": []})

        assert resp.status == "error"
