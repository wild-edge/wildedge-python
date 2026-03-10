from __future__ import annotations

import random
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from wildedge import constants


@dataclass
class ReservoirStats:
    total_inference_events_seen: int = 0
    total_inference_events_sent: int = 0
    priority_seen: int = 0
    priority_sent: int = 0


class InferenceReservoir:
    """
    Two-stratum per-model inference reservoir sampler (Algorithm R).

    Stratum A — priority (always retain, sample_rate omitted):
      1. success=False
      2. output_meta.avg_confidence < low_confidence_threshold
      3. output_meta.avg_token_entropy > high_entropy_threshold
      4. priority_fn(event) returns True (replaces built-ins when set)

    Stratum B — background (Algorithm R, sample_rate set at flush time).
    """

    def __init__(
        self,
        reservoir_size: int = constants.DEFAULT_RESERVOIR_SIZE,
        low_confidence_threshold: float = constants.DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        high_entropy_threshold: float = constants.DEFAULT_HIGH_ENTROPY_THRESHOLD,
        low_confidence_slots_pct: float = constants.DEFAULT_LOW_CONFIDENCE_SLOTS_PCT,
        priority_fn: Callable[[dict], bool] | None = None,
    ) -> None:
        self._reservoir_size = reservoir_size
        self._low_confidence_threshold = low_confidence_threshold
        self._high_entropy_threshold = high_entropy_threshold
        self._background_capacity = max(
            1, reservoir_size - int(reservoir_size * low_confidence_slots_pct)
        )
        self._priority_fn = priority_fn

        # Stratum A: unbounded — all priority events are always retained
        self._stratum_a: list[dict] = []
        self._stratum_a_seen: int = 0

        # Stratum B: Algorithm R reservoir
        self._stratum_b: list[dict] = []
        self._stratum_b_seen: int = 0

        self._lock = threading.Lock()

    def add(self, event: dict) -> None:
        with self._lock:
            if self._is_priority(event):
                self._stratum_a.append(event)
                self._stratum_a_seen += 1
            else:
                self._stratum_b_seen += 1
                if len(self._stratum_b) < self._background_capacity:
                    self._stratum_b.append(event)
                else:
                    j = random.randint(0, self._stratum_b_seen - 1)
                    if j < self._background_capacity:
                        self._stratum_b[j] = event

    def snapshot(self) -> tuple[list[dict[str, Any]], ReservoirStats]:
        """Atomically drain and reset. Annotates Stratum B events with sample_rate."""
        with self._lock:
            stratum_a = self._stratum_a
            stratum_b = self._stratum_b
            a_seen = self._stratum_a_seen
            b_seen = self._stratum_b_seen

            self._stratum_a = []
            self._stratum_b = []
            self._stratum_a_seen = 0
            self._stratum_b_seen = 0

        # Annotate Stratum B events with sample_rate (omit when 1.0 i.e. warm-up)
        if b_seen > 0:
            b_rate = len(stratum_b) / b_seen
            if b_rate < 1.0:
                for ev in stratum_b:
                    ev["sample_rate"] = b_rate

        stats = ReservoirStats(
            total_inference_events_seen=a_seen + b_seen,
            total_inference_events_sent=len(stratum_a) + len(stratum_b),
            priority_seen=a_seen,
            priority_sent=len(stratum_a),
        )

        return stratum_a + stratum_b, stats

    def size(self) -> int:
        with self._lock:
            return len(self._stratum_a) + len(self._stratum_b)

    def _is_priority(self, event: dict) -> bool:
        if self._priority_fn is not None:
            return self._priority_fn(event)

        inference = event.get("inference", {})

        if not inference.get("success", True):
            return True

        output_meta = inference.get("output_meta", {})
        if isinstance(output_meta, dict):
            avg_confidence = output_meta.get("avg_confidence")
            if (
                avg_confidence is not None
                and avg_confidence < self._low_confidence_threshold
            ):
                return True

            avg_token_entropy = output_meta.get("avg_token_entropy")
            if (
                avg_token_entropy is not None
                and avg_token_entropy > self._high_entropy_threshold
            ):
                return True

        return False


class ReservoirRegistry:
    """Thread-safe registry of per-model InferenceReservoirs."""

    def __init__(
        self,
        reservoir_size: int = constants.DEFAULT_RESERVOIR_SIZE,
        low_confidence_threshold: float = constants.DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        high_entropy_threshold: float = constants.DEFAULT_HIGH_ENTROPY_THRESHOLD,
        low_confidence_slots_pct: float = constants.DEFAULT_LOW_CONFIDENCE_SLOTS_PCT,
        priority_fn: Callable[[dict], bool] | None = None,
    ) -> None:
        self._reservoir_kwargs: dict[str, Any] = {
            "reservoir_size": reservoir_size,
            "low_confidence_threshold": low_confidence_threshold,
            "high_entropy_threshold": high_entropy_threshold,
            "low_confidence_slots_pct": low_confidence_slots_pct,
            "priority_fn": priority_fn,
        }
        self._reservoirs: dict[str, InferenceReservoir] = {}
        self._lock = threading.Lock()

    def get_or_create(self, model_id: str) -> InferenceReservoir:
        with self._lock:
            if model_id not in self._reservoirs:
                self._reservoirs[model_id] = InferenceReservoir(
                    **self._reservoir_kwargs
                )
            return self._reservoirs[model_id]

    def has_events(self) -> bool:
        """Return True if any reservoir holds at least one event."""
        with self._lock:
            return any(r.size() > 0 for r in self._reservoirs.values())

    def snapshot_all(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Snapshot every reservoir atomically. Returns:
          - combined list of all inference events (with sample_rate where < 1.0)
          - sampling envelope dict for the batch (priority_thresholds + per-model stats)
        """
        with self._lock:
            model_ids = list(self._reservoirs.keys())

        all_events: list[dict[str, Any]] = []
        sampling: dict[str, Any] = {
            "priority_thresholds": {
                "low_confidence": self._reservoir_kwargs["low_confidence_threshold"],
                "high_entropy": self._reservoir_kwargs["high_entropy_threshold"],
            }
        }

        for model_id in model_ids:
            reservoir = self._reservoirs[model_id]
            events, stats = reservoir.snapshot()
            if stats.total_inference_events_seen == 0:
                continue
            all_events.extend(events)
            sampling[model_id] = {
                "total_inference_events_seen": stats.total_inference_events_seen,
                "total_inference_events_sent": stats.total_inference_events_sent,
                "priority_seen": stats.priority_seen,
                "priority_sent": stats.priority_sent,
            }

        return all_events, sampling
