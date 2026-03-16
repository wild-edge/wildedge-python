"""Tests for the background consumer."""

import os
from unittest.mock import MagicMock, patch

import pytest

from wildedge import constants
from wildedge.consumer import Consumer
from wildedge.device import DeviceInfo
from wildedge.queue import EventQueue
from wildedge.transmitter import IngestResponse, TransmitError, Transmitter


def make_device() -> DeviceInfo:
    return DeviceInfo(
        app_version="1.0.0",
        device_id="device-id",
        device_type="linux",
    )


def make_success_response(n_events: int = 1) -> IngestResponse:
    return IngestResponse(
        status="accepted",
        batch_id="batch-1",
        events_accepted=n_events,
        events_rejected=0,
    )


class TestConsumerFlush:
    def setup_method(self):
        self._consumers: list[Consumer] = []

    def teardown_method(self):
        for c in self._consumers:
            c.stop()

    def _make_consumer(self, queue, transmitter, **kwargs) -> Consumer:
        c = Consumer(
            queue=queue,
            transmitter=transmitter,
            device=make_device(),
            get_models=lambda: {},
            session_id="sess-1",
            **kwargs,
        )
        self._consumers.append(c)
        return c

    def test_flush_drains_queue(self):
        queue = EventQueue(max_size=100)
        for i in range(3):
            queue.add({"event_id": str(i), "event_type": "inference", "model_id": "m"})

        mock_transmitter = MagicMock(spec=Transmitter)
        mock_transmitter.send.return_value = make_success_response(3)

        consumer = self._make_consumer(
            queue, mock_transmitter, batch_size=10, flush_interval_sec=60.0
        )
        consumer.flush(timeout=2.0)
        assert queue.length() == 0

    def test_flush_calls_transmitter(self):
        queue = EventQueue(max_size=100)
        queue.add({"event_id": "e1", "event_type": "inference", "model_id": "m"})

        mock_transmitter = MagicMock(spec=Transmitter)
        mock_transmitter.send.return_value = make_success_response(1)

        consumer = self._make_consumer(queue, mock_transmitter)
        consumer.flush(timeout=2.0)
        assert mock_transmitter.send.called

    def test_transmit_error_keeps_events_in_queue(self):
        queue = EventQueue(max_size=100)
        queue.add({"event_id": "e1", "event_type": "inference", "model_id": "m"})

        mock_transmitter = MagicMock(spec=Transmitter)
        mock_transmitter.send.side_effect = TransmitError("Server error")

        consumer = self._make_consumer(queue, mock_transmitter)
        consumer.drain_once()  # Should not raise, events stay in queue
        assert queue.length() == 1

    def test_400_discards_events(self):
        queue = EventQueue(max_size=100)
        queue.add({"event_id": "e1", "event_type": "inference", "model_id": "m"})

        mock_transmitter = MagicMock(spec=Transmitter)
        mock_transmitter.send.return_value = IngestResponse(
            status="rejected",
            batch_id="b-1",
            events_accepted=0,
            events_rejected=1,
        )

        consumer = self._make_consumer(queue, mock_transmitter)
        consumer.drain_once()
        assert queue.length() == 0  # Events discarded on 400

    def test_permanent_error_discards_events(self):
        queue = EventQueue(max_size=100)
        queue.add({"event_id": "e1", "event_type": "inference", "model_id": "m"})

        mock_transmitter = MagicMock(spec=Transmitter)
        mock_transmitter.send.return_value = IngestResponse(
            status="error",
            batch_id="b-1",
            events_accepted=0,
            events_rejected=1,
        )

        consumer = self._make_consumer(queue, mock_transmitter)
        consumer.drain_once()
        assert queue.length() == 0  # Permanent transmitter error is discarded

    def test_permanent_error_persists_dead_letter_and_calls_callback(self):
        queue = EventQueue(max_size=100)
        queue.add({"event_id": "e1", "event_type": "inference", "model_id": "m"})
        dead_letter_store = MagicMock()
        on_failure = MagicMock()

        mock_transmitter = MagicMock(spec=Transmitter)
        mock_transmitter.send.return_value = IngestResponse(
            status="error",
            batch_id="b-1",
            events_accepted=0,
            events_rejected=1,
        )

        consumer = self._make_consumer(
            queue,
            mock_transmitter,
            dead_letter_store=dead_letter_store,
            on_delivery_failure=on_failure,
        )
        consumer.drain_once()

        assert queue.length() == 0
        assert dead_letter_store.write.called
        reason = dead_letter_store.write.call_args.kwargs["reason"]
        assert reason == "permanent_error"
        assert on_failure.called
        assert on_failure.call_args_list[0].args == ("permanent_error", 1, 0)

    def test_max_event_age_drops_stale_events_without_transmit(self):
        queue = EventQueue(max_size=100)
        queue.add(
            {
                "event_id": "e1",
                "event_type": "inference",
                "model_id": "m",
                "__we_first_queued_at": 1.0,
            }
        )
        dead_letter_store = MagicMock()

        mock_transmitter = MagicMock(spec=Transmitter)

        consumer = self._make_consumer(
            queue,
            mock_transmitter,
            max_event_age_sec=0.01,
            dead_letter_store=dead_letter_store,
        )
        consumer.drain_once()

        assert queue.length() == 0
        mock_transmitter.send.assert_not_called()
        assert dead_letter_store.write.called

    def test_batch_includes_model_registry(self):
        queue = EventQueue(max_size=100)
        queue.add({"event_id": "e1", "event_type": "inference", "model_id": "my-model"})

        sent_batches = []
        mock_transmitter = MagicMock(spec=Transmitter)
        mock_transmitter.send.side_effect = lambda b: (
            sent_batches.append(b) or make_success_response(1)
        )

        models = {"my-model": {"model_name": "test", "model_version": "1.0"}}
        consumer = Consumer(
            queue=queue,
            transmitter=mock_transmitter,
            device=make_device(),
            get_models=lambda: models,
            session_id="sess-1",
        )
        self._consumers.append(consumer)

        consumer.drain_once()
        assert sent_batches[0]["models"] == models

    def test_drain_once_returns_false_when_empty(self):
        queue = EventQueue(max_size=100)
        mock_transmitter = MagicMock(spec=Transmitter)

        consumer = self._make_consumer(queue, mock_transmitter)
        result = consumer.drain_once()
        assert result is False
        mock_transmitter.send.assert_not_called()

    def test_flush_does_not_tight_loop_on_transmit_error(self, monkeypatch):
        monkeypatch.setattr(Consumer, "run", lambda self: None)
        queue = EventQueue(max_size=100)
        queue.add({"event_id": "e1", "event_type": "inference", "model_id": "m"})
        mock_transmitter = MagicMock(spec=Transmitter)
        mock_transmitter.send.side_effect = TransmitError("Network error")

        consumer = self._make_consumer(queue, mock_transmitter)
        consumer.flush(timeout=0.2)
        assert mock_transmitter.send.call_count == 1

    def test_next_retry_delay_scales_and_caps(self):
        queue = EventQueue(max_size=100)
        mock_transmitter = MagicMock(spec=Transmitter)
        consumer = self._make_consumer(queue, mock_transmitter)

        delay, next_backoff = consumer.next_retry_delay(1.0, jitter=False)
        assert delay == 1.0
        assert next_backoff == 2.0

        delay, next_backoff = consumer.next_retry_delay(1.0, jitter=False, max_wait=0.3)
        assert delay == 0.3
        assert next_backoff == 2.0

    def test_next_retry_delay_applies_jitter(self, monkeypatch):
        queue = EventQueue(max_size=100)
        mock_transmitter = MagicMock(spec=Transmitter)
        consumer = self._make_consumer(queue, mock_transmitter)

        monkeypatch.setattr("wildedge.consumer.random.uniform", lambda a, b: 0.2)
        delay, _ = consumer.next_retry_delay(1.0, jitter=True)
        assert delay == 1.2

    def test_run_uses_shared_retry_delay(self, monkeypatch):
        original_run = Consumer.run
        monkeypatch.setattr(Consumer, "run", lambda self: None)
        queue = EventQueue(max_size=100)
        queue.add({"event_id": "e1", "event_type": "inference", "model_id": "m"})
        mock_transmitter = MagicMock(spec=Transmitter)
        consumer = self._make_consumer(queue, mock_transmitter)
        consumer.run = original_run.__get__(consumer, Consumer)

        class StopControl:
            def __init__(self):
                self.flag = False
                self.wait_calls: list[float] = []

            def is_set(self) -> bool:
                return self.flag

            def wait(self, timeout=None) -> None:
                self.wait_calls.append(timeout)
                self.flag = True

            def set(self) -> None:
                self.flag = True

        stop_control = StopControl()
        consumer.stop_event = stop_control
        consumer.drain_once = lambda: False

        called = {"count": 0}

        def fake_next_retry_delay(backoff, *, jitter, max_wait=None):
            called["count"] += 1
            assert backoff == 1.0
            assert jitter is True
            assert max_wait is None
            return 0.42, 2.0

        consumer.next_retry_delay = fake_next_retry_delay
        consumer.run()
        assert called["count"] == 1
        assert stop_control.wait_calls[0] == 0.42

    def test_flush_uses_shared_retry_delay(self, monkeypatch):
        monkeypatch.setattr(Consumer, "run", lambda self: None)
        queue = EventQueue(max_size=100)
        queue.add({"event_id": "e1", "event_type": "inference", "model_id": "m"})
        mock_transmitter = MagicMock(spec=Transmitter)
        mock_transmitter.send.side_effect = TransmitError("Network error")

        consumer = self._make_consumer(queue, mock_transmitter)
        calls = {"count": 0}

        def fake_next_retry_delay(backoff, *, jitter, max_wait=None):
            calls["count"] += 1
            assert jitter is True
            assert max_wait is not None
            return max_wait, backoff

        consumer.next_retry_delay = fake_next_retry_delay
        consumer.flush(timeout=0.15)
        assert calls["count"] >= 1

    def test_close_default_is_best_effort_non_blocking(self, monkeypatch):
        monkeypatch.setattr(Consumer, "run", lambda self: None)
        queue = EventQueue(max_size=100)
        queue.add({"event_id": "e1", "event_type": "inference", "model_id": "m"})
        mock_transmitter = MagicMock(spec=Transmitter)
        mock_transmitter.send.side_effect = TransmitError("Network error")

        consumer = self._make_consumer(queue, mock_transmitter)
        consumer.close()
        assert mock_transmitter.send.call_count == 0
        assert queue.length() == 1

    def test_close_with_timeout_attempts_flush(self, monkeypatch):
        monkeypatch.setattr(Consumer, "run", lambda self: None)
        queue = EventQueue(max_size=100)
        queue.add({"event_id": "e1", "event_type": "inference", "model_id": "m"})
        mock_transmitter = MagicMock(spec=Transmitter)
        mock_transmitter.send.side_effect = TransmitError("Network error")

        consumer = self._make_consumer(queue, mock_transmitter)
        consumer.close(timeout=0.15)
        assert mock_transmitter.send.call_count >= 1

    def test_atexit_registers_shutdown_flush_budget(self, monkeypatch):
        registered = {}

        def fake_register(fn, *args, **kwargs):
            registered["fn"] = fn
            registered["args"] = args
            registered["kwargs"] = kwargs

        monkeypatch.setattr("wildedge.consumer.atexit.register", fake_register)
        monkeypatch.setattr(Consumer, "run", lambda self: None)

        queue = EventQueue(max_size=100)
        mock_transmitter = MagicMock(spec=Transmitter)
        consumer = self._make_consumer(queue, mock_transmitter)
        assert registered["fn"] == consumer.flush
        assert registered["args"] == (constants.DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC,)


class TestConsumerForkSafety:
    def setup_method(self):
        self._consumers: list[Consumer] = []

    def teardown_method(self):
        for c in self._consumers:
            c.stop()

    def _make_consumer(self, queue, transmitter, **kwargs) -> Consumer:
        c = Consumer(
            queue=queue,
            transmitter=transmitter,
            device=DeviceInfo(app_version="1.0", device_id="d", device_type="linux"),
            get_models=lambda: {},
            session_id="sess-fork",
            **kwargs,
        )
        self._consumers.append(c)
        return c

    def test_pause_stops_thread(self, monkeypatch):
        monkeypatch.setattr(Consumer, "run", lambda self: None)
        queue = EventQueue(max_size=100)
        mock_tx = MagicMock(spec=Transmitter)
        consumer = self._make_consumer(queue, mock_tx)

        original_thread = consumer.thread
        consumer._pause()

        assert consumer.stop_event.is_set()
        assert not original_thread.is_alive()
        # stopped is reset to False so _resume() can start a new thread
        assert consumer.stopped is False

    def test_resume_creates_fresh_thread(self, monkeypatch):
        monkeypatch.setattr(Consumer, "run", lambda self: None)
        queue = EventQueue(max_size=100)
        mock_tx = MagicMock(spec=Transmitter)
        consumer = self._make_consumer(queue, mock_tx)

        original_thread = consumer.thread
        consumer._pause()
        consumer._resume()

        assert consumer.thread is not original_thread
        # thread was started (ident is set) even though the no-op run() exits fast
        assert consumer.thread.ident is not None
        assert not consumer.stop_event.is_set()

    def test_resume_resets_backoff_and_snapshot(self, monkeypatch):
        monkeypatch.setattr(Consumer, "run", lambda self: None)
        queue = EventQueue(max_size=100)
        mock_tx = MagicMock(spec=Transmitter)
        consumer = self._make_consumer(queue, mock_tx)

        consumer.backoff = constants.BACKOFF_MAX
        consumer._held_snapshot = ([], {})

        consumer._pause()
        consumer._resume()

        assert consumer.backoff == constants.BACKOFF_MIN
        assert consumer._held_snapshot is None

    def test_flush_is_noop_after_pause_before_resume(self, monkeypatch):
        """flush() on a pre-fork-stopped consumer with empty queue returns immediately."""
        monkeypatch.setattr(Consumer, "run", lambda self: None)
        queue = EventQueue(max_size=100)
        mock_tx = MagicMock(spec=Transmitter)
        consumer = self._make_consumer(queue, mock_tx)

        consumer._pause()
        # stopped is False (reset by _pause) and queue is empty, so flush
        # calls drain_once which returns False immediately; no transmit calls.
        consumer.flush(timeout=0.1)
        mock_tx.send.assert_not_called()

    def test_resume_registers_atexit(self, monkeypatch):
        """_resume() registers a new atexit flush callback for the fresh thread."""
        monkeypatch.setattr(Consumer, "run", lambda self: None)
        queue = EventQueue(max_size=100)
        mock_tx = MagicMock(spec=Transmitter)
        consumer = self._make_consumer(queue, mock_tx)

        registered = []
        monkeypatch.setattr(
            "wildedge.consumer.atexit.register",
            lambda fn, *a: registered.append((fn, a)),
        )

        consumer._pause()
        consumer._resume()

        assert any(fn == consumer.flush for fn, _ in registered)


class TestForkRegistration:
    @pytest.mark.skipif(
        not hasattr(os, "register_at_fork"),
        reason="os.register_at_fork not available on Windows",
    )
    def test_register_at_fork_wires_pause_and_resume(self, monkeypatch):
        """WildEdge.__init__ registers _pause and _resume via os.register_at_fork."""
        from wildedge.client import WildEdge

        registered = {}

        def fake_register_at_fork(*, before, after_in_child, after_in_parent):
            registered["before"] = before
            registered["after_in_child"] = after_in_child
            registered["after_in_parent"] = after_in_parent

        with (
            patch.object(os, "register_at_fork", fake_register_at_fork),
            patch(
                "wildedge.client.detect_device",
                return_value=DeviceInfo(device_id="d", device_type="linux"),
            ),
            patch("wildedge.client.Consumer") as mock_consumer_cls,
        ):
            consumer_instance = mock_consumer_cls.return_value
            WildEdge(dsn="https://secret@ingest.wildedge.dev/key")

        assert registered["before"] is consumer_instance._pause
        assert registered["after_in_child"] is consumer_instance._resume
        assert registered["after_in_parent"] is consumer_instance._resume

    def test_register_at_fork_noop_on_windows(self, monkeypatch):
        """When os.register_at_fork is absent (Windows), __init__ must not raise."""
        from wildedge.client import WildEdge

        monkeypatch.delattr(os, "register_at_fork", raising=False)

        with (
            patch(
                "wildedge.client.detect_device",
                return_value=DeviceInfo(device_id="d", device_type="linux"),
            ),
            patch("wildedge.client.Consumer", return_value=MagicMock()),
        ):
            client = WildEdge(dsn="https://secret@ingest.wildedge.dev/key")
            client.consumer.stop = MagicMock()
