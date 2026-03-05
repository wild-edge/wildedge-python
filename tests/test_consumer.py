"""Tests for the background consumer."""

from unittest.mock import MagicMock

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
