"""Tests for InferenceReservoir and ReservoirRegistry."""

from __future__ import annotations

import threading

from wildedge.reservoir import InferenceReservoir, ReservoirRegistry, ReservoirStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inference_event(
    model_id: str = "m1",
    success: bool = True,
    avg_confidence: float | None = None,
    avg_token_entropy: float | None = None,
) -> dict:
    output_meta: dict = {}
    if avg_confidence is not None:
        output_meta["avg_confidence"] = avg_confidence
    if avg_token_entropy is not None:
        output_meta["avg_token_entropy"] = avg_token_entropy

    inference: dict = {"success": success}
    if output_meta:
        inference["output_meta"] = output_meta

    return {
        "event_type": "inference",
        "model_id": model_id,
        "inference": inference,
    }


# ---------------------------------------------------------------------------
# Stratum A guarantee
# ---------------------------------------------------------------------------


def test_stratum_a_success_false_always_retained():
    r = InferenceReservoir(reservoir_size=5)
    for _ in range(20):
        r.add(make_inference_event(success=False))
    events, stats = r.snapshot()
    assert stats.priority_seen == 20
    assert stats.priority_sent == 20
    assert stats.total_inference_events_sent == 20
    assert all("sample_rate" not in e for e in events)


def test_stratum_a_low_confidence_always_retained():
    r = InferenceReservoir(reservoir_size=5, low_confidence_threshold=0.5)
    for _ in range(20):
        r.add(make_inference_event(avg_confidence=0.1))
    events, stats = r.snapshot()
    assert stats.priority_seen == 20
    assert stats.priority_sent == 20
    assert all("sample_rate" not in e for e in events)


def test_stratum_a_high_entropy_always_retained():
    r = InferenceReservoir(reservoir_size=5, high_entropy_threshold=2.0)
    for _ in range(20):
        r.add(make_inference_event(avg_token_entropy=3.5))
    events, stats = r.snapshot()
    assert stats.priority_seen == 20
    assert stats.priority_sent == 20
    assert all("sample_rate" not in e for e in events)


def test_stratum_a_threshold_boundary():
    r = InferenceReservoir(
        reservoir_size=10,
        low_confidence_threshold=0.5,
        high_entropy_threshold=2.0,
    )
    # Exactly at threshold: not priority
    r.add(make_inference_event(avg_confidence=0.5))
    r.add(make_inference_event(avg_token_entropy=2.0))
    # Just inside threshold: priority
    r.add(make_inference_event(avg_confidence=0.49))
    r.add(make_inference_event(avg_token_entropy=2.01))

    _, stats = r.snapshot()
    assert stats.priority_seen == 2
    assert stats.priority_sent == 2


# ---------------------------------------------------------------------------
# priority_fn override
# ---------------------------------------------------------------------------


def test_priority_fn_replaces_builtin_signals():
    # Built-in signals would make these priority, but priority_fn always returns False
    r = InferenceReservoir(
        reservoir_size=3,
        priority_fn=lambda e: False,
    )
    for _ in range(10):
        r.add(make_inference_event(success=False))
    _, stats = r.snapshot()
    assert stats.priority_seen == 0


def test_priority_fn_can_promote_to_stratum_a():
    def my_fn(event: dict) -> bool:
        return event.get("inference", {}).get("success", True) is True

    r = InferenceReservoir(reservoir_size=3, priority_fn=my_fn)
    for _ in range(20):
        r.add(make_inference_event(success=True))
    _, stats = r.snapshot()
    assert stats.priority_seen == 20
    assert stats.priority_sent == 20


# ---------------------------------------------------------------------------
# Warm-up: no sample_rate while seen < capacity
# ---------------------------------------------------------------------------


def test_warmup_no_sample_rate():
    r = InferenceReservoir(reservoir_size=10, low_confidence_slots_pct=0.2)
    # background_capacity = 10 - int(10 * 0.2) = 8
    for _ in range(8):  # exactly fills stratum B during warm-up
        r.add(make_inference_event(avg_confidence=0.9))
    events, stats = r.snapshot()
    assert all("sample_rate" not in e for e in events)
    assert stats.total_inference_events_sent == 8


# ---------------------------------------------------------------------------
# Post-warm-up: sample_rate set on Stratum B events only
# ---------------------------------------------------------------------------


def test_sample_rate_set_post_warmup():
    b_cap = 4
    r = InferenceReservoir(reservoir_size=5, low_confidence_slots_pct=0.2)
    # background_capacity = 5 - int(5 * 0.2) = 4
    total_bg = 20
    for _ in range(total_bg):
        r.add(make_inference_event(avg_confidence=0.9))
    events, stats = r.snapshot()

    assert stats.total_inference_events_seen == total_bg
    assert stats.total_inference_events_sent == b_cap
    assert stats.priority_seen == 0

    expected_rate = b_cap / total_bg
    for e in events:
        assert "sample_rate" in e
        assert abs(e["sample_rate"] - expected_rate) < 1e-9


def test_no_sample_rate_on_stratum_a_events_post_warmup():
    r = InferenceReservoir(reservoir_size=5, low_confidence_slots_pct=0.2)
    for _ in range(20):
        r.add(make_inference_event(avg_confidence=0.1))  # priority
    for _ in range(20):
        r.add(make_inference_event(avg_confidence=0.9))  # background
    events, stats = r.snapshot()

    priority_events = [e for e in events if "sample_rate" not in e]
    background_events = [e for e in events if "sample_rate" in e]

    assert len(priority_events) == 20  # all priority retained
    assert stats.priority_sent == 20
    assert len(background_events) <= 4  # background capacity
    for e in background_events:
        assert e["sample_rate"] < 1.0


# ---------------------------------------------------------------------------
# snapshot() atomically resets
# ---------------------------------------------------------------------------


def test_snapshot_resets_counters():
    r = InferenceReservoir(reservoir_size=10)
    for _ in range(5):
        r.add(make_inference_event())
    r.snapshot()

    assert r.size() == 0
    _, stats = r.snapshot()
    assert stats.total_inference_events_seen == 0
    assert stats.total_inference_events_sent == 0
    assert stats.priority_seen == 0
    assert stats.priority_sent == 0


def test_snapshot_empty_reservoir():
    r = InferenceReservoir()
    events, stats = r.snapshot()
    assert events == []
    assert stats == ReservoirStats()


# ---------------------------------------------------------------------------
# Algorithm R statistical distribution
# ---------------------------------------------------------------------------


def test_algorithm_r_uniform_distribution():
    """Each slot should be selected with roughly equal probability."""
    n_slots = 4
    n_total = 400
    n_trials = 200
    slot_counts: dict[int, int] = {i: 0 for i in range(n_slots)}

    # background_capacity = 5 - 1 = 4 with low_confidence_slots_pct=0.2, size=5
    for _ in range(n_trials):
        r = InferenceReservoir(reservoir_size=5, low_confidence_slots_pct=0.2)
        for i in range(n_total):
            ev = make_inference_event(avg_confidence=0.9)
            ev["idx"] = i
            r.add(ev)
        events, _ = r.snapshot()
        for e in events:
            bucket = e["idx"] % n_slots
            slot_counts[bucket] += 1

    total = sum(slot_counts.values())
    expected = total / n_slots
    for count in slot_counts.values():
        # Allow 30% deviation from expected (loose but deterministic)
        assert abs(count - expected) / expected < 0.3


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_add_and_snapshot():
    r = InferenceReservoir(reservoir_size=50)
    errors: list[Exception] = []

    def producer():
        try:
            for _ in range(500):
                r.add(make_inference_event(avg_confidence=0.9))
        except Exception as exc:
            errors.append(exc)

    def snapshotter():
        try:
            for _ in range(10):
                r.snapshot()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=producer) for _ in range(4)]
    threads += [threading.Thread(target=snapshotter) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


# ---------------------------------------------------------------------------
# ReservoirRegistry
# ---------------------------------------------------------------------------


def test_registry_get_or_create_is_idempotent():
    reg = ReservoirRegistry()
    r1 = reg.get_or_create("model-a")
    r2 = reg.get_or_create("model-a")
    assert r1 is r2


def test_registry_snapshot_all_sampling_envelope():
    reg = ReservoirRegistry(
        reservoir_size=10,
        low_confidence_threshold=0.4,
        high_entropy_threshold=1.5,
    )
    for _ in range(3):
        reg.get_or_create("m1").add(make_inference_event("m1", avg_confidence=0.1))
    for _ in range(2):
        reg.get_or_create("m2").add(make_inference_event("m2", avg_confidence=0.9))

    events, sampling = reg.snapshot_all()

    assert len(events) == 5
    assert sampling["priority_thresholds"]["low_confidence"] == 0.4
    assert sampling["priority_thresholds"]["high_entropy"] == 1.5

    assert sampling["m1"]["total_inference_events_seen"] == 3
    assert sampling["m1"]["priority_seen"] == 3
    assert sampling["m1"]["priority_sent"] == 3

    assert sampling["m2"]["total_inference_events_seen"] == 2
    assert sampling["m2"]["priority_seen"] == 0


def test_registry_snapshot_all_excludes_empty_models():
    reg = ReservoirRegistry()
    reg.get_or_create("m1")  # never receives events
    reg.get_or_create("m2").add(make_inference_event("m2"))

    _, sampling = reg.snapshot_all()
    assert "m1" not in sampling
    assert "m2" in sampling


def test_registry_has_events():
    reg = ReservoirRegistry()
    assert not reg.has_events()
    reg.get_or_create("m1").add(make_inference_event("m1"))
    assert reg.has_events()
    reg.snapshot_all()
    assert not reg.has_events()
