"""Tests for EventQueue."""

import threading

import pytest

from wildedge.queue import EventQueue, FifoFullError, QueuePolicy


def make_event(n: int) -> dict:
    return {"event_id": str(n), "event_type": "inference"}


class TestEventQueue:
    def test_add_and_peek(self):
        q = EventQueue(max_size=10)
        q.add(make_event(1))
        assert q.peek() == make_event(1)
        assert q.length() == 1

    def test_peek_empty_returns_none(self):
        q = EventQueue()
        assert q.peek() is None

    def test_peek_many(self):
        q = EventQueue(max_size=10)
        for i in range(5):
            q.add(make_event(i))
        result = q.peek_many(3)
        assert len(result) == 3
        assert result[0] == make_event(0)

    def test_peek_many_more_than_available(self):
        q = EventQueue(max_size=10)
        q.add(make_event(1))
        result = q.peek_many(100)
        assert len(result) == 1

    def test_remove_first(self):
        q = EventQueue(max_size=10)
        q.add(make_event(1))
        q.add(make_event(2))
        q.remove_first()
        assert q.length() == 1
        assert q.peek() == make_event(2)

    def test_remove_first_n(self):
        q = EventQueue(max_size=10)
        for i in range(5):
            q.add(make_event(i))
        q.remove_first_n(3)
        assert q.length() == 2
        assert q.peek() == make_event(3)

    def test_remove_first_n_more_than_available(self):
        q = EventQueue(max_size=3)
        for i in range(3):
            q.add(make_event(i))
        q.remove_first_n(100)  # Should not raise
        assert q.length() == 0

    def test_length(self):
        q = EventQueue(max_size=10)
        assert q.length() == 0
        q.add(make_event(1))
        assert q.length() == 1

    def test_strict_policy_raises_when_full(self):
        q = EventQueue(max_size=2, policy=QueuePolicy.STRICT)
        q.add(make_event(1))
        q.add(make_event(2))
        with pytest.raises(FifoFullError):
            q.add(make_event(3))

    def test_opportunistic_policy_drops_oldest(self):
        q = EventQueue(max_size=2, policy=QueuePolicy.OPPORTUNISTIC)
        q.add(make_event(1))
        q.add(make_event(2))
        q.add(make_event(3))  # Should drop event 1
        assert q.length() == 2
        assert q.peek() == make_event(2)

    def test_fifo_ordering(self):
        q = EventQueue(max_size=10)
        for i in range(5):
            q.add(make_event(i))
        result = q.peek_many(5)
        assert [e["event_id"] for e in result] == ["0", "1", "2", "3", "4"]

    def test_thread_safe_concurrent_add(self):
        q = EventQueue(max_size=1000, policy=QueuePolicy.OPPORTUNISTIC)
        errors = []

        def add_events():
            try:
                for i in range(50):
                    q.add({"event_id": str(i)})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_events) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert q.length() <= 1000

    def test_persistent_queue_rehydrates_from_disk(self, tmp_path):
        q1 = EventQueue(
            max_size=10,
            policy=QueuePolicy.OPPORTUNISTIC,
            persist_to_disk=True,
            disk_dir=str(tmp_path),
        )
        q1.add(make_event(1))
        q1.add(make_event(2))

        q2 = EventQueue(
            max_size=10,
            policy=QueuePolicy.OPPORTUNISTIC,
            persist_to_disk=True,
            disk_dir=str(tmp_path),
        )
        assert q2.length() == 2
        assert q2.peek() == make_event(1)

    def test_persistent_queue_remove_deletes_files(self, tmp_path):
        q = EventQueue(
            max_size=10,
            policy=QueuePolicy.OPPORTUNISTIC,
            persist_to_disk=True,
            disk_dir=str(tmp_path),
        )
        q.add(make_event(1))
        q.add(make_event(2))
        assert len(list(tmp_path.glob("*.json"))) == 2
        q.remove_first()
        assert len(list(tmp_path.glob("*.json"))) == 1
