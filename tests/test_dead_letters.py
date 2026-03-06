from __future__ import annotations

from wildedge.dead_letters import DeadLetterStore


def test_dead_letter_store_writes_batch_file(tmp_path):
    store = DeadLetterStore(
        enabled=True,
        directory=str(tmp_path),
        max_batches=10,
    )
    store.write(reason="test", events=[{"event_id": "e1"}], batch_id="b1")
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1


def test_dead_letter_store_enforces_max_batches(tmp_path):
    store = DeadLetterStore(
        enabled=True,
        directory=str(tmp_path),
        max_batches=2,
    )
    store.write(reason="test", events=[{"event_id": "e1"}], batch_id="b1")
    store.write(reason="test", events=[{"event_id": "e2"}], batch_id="b2")
    store.write(reason="test", events=[{"event_id": "e3"}], batch_id="b3")
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 2


def test_dead_letter_store_disabled_no_files(tmp_path):
    store = DeadLetterStore(
        enabled=False,
        directory=str(tmp_path),
        max_batches=10,
    )
    store.write(reason="test", events=[{"event_id": "e1"}], batch_id="b1")
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 0
