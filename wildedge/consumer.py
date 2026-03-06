from __future__ import annotations

import atexit
import random
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from wildedge import constants
from wildedge.batch import build_batch
from wildedge.dead_letters import DeadLetterStore
from wildedge.logging import logger
from wildedge.queue import EventQueue
from wildedge.transmitter import TransmitError, Transmitter

if TYPE_CHECKING:
    from wildedge.device import DeviceInfo


class Consumer:
    """Pulls events from the queue, batches, and transmits them."""

    def __init__(
        self,
        queue: EventQueue,
        transmitter: Transmitter,
        device: DeviceInfo,
        get_models: Callable[[], dict[str, dict]],
        session_id: str,
        batch_size: int = 10,
        flush_interval_sec: float = 60.0,
        debug: bool = False,
        max_event_age_sec: float = constants.DEFAULT_MAX_EVENT_AGE_SEC,
        dead_letter_store: DeadLetterStore | None = None,
        on_delivery_failure: Callable[[str, int, int], None] | None = None,
    ):
        self.queue = queue
        self.transmitter = transmitter
        self.device = device
        self.get_models = get_models
        self.session_id = session_id
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        self.debug = debug
        self.max_event_age_sec = max_event_age_sec
        self.dead_letter_store = dead_letter_store
        self.on_delivery_failure = on_delivery_failure

        self.stop_event = threading.Event()
        self.stopped = False
        self.backoff = constants.BACKOFF_MIN
        self.created_at = datetime.now(timezone.utc)

        self.thread = threading.Thread(
            target=self.run, daemon=True, name="wildedge-consumer"
        )
        self.thread.start()
        atexit.register(self.flush, constants.DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC)

    def run(self) -> None:
        last_flush = time.monotonic()
        while not self.stop_event.is_set():
            now = time.monotonic()
            time_since_flush = now - last_flush
            force_flush = time_since_flush >= self.flush_interval_sec

            if self.queue.length() > 0 or force_flush:
                sent = self.drain_once()
                if sent:
                    last_flush = time.monotonic()
                    self.backoff = constants.BACKOFF_MIN
                else:
                    wait_s, self.backoff = self.next_retry_delay(
                        self.backoff,
                        jitter=True,
                    )
                    self.stop_event.wait(timeout=wait_s)
            else:
                self.stop_event.wait(timeout=constants.IDLE_POLL_INTERVAL)

    def next_retry_delay(
        self,
        backoff: float,
        *,
        jitter: bool,
        max_wait: float | None = None,
    ) -> tuple[float, float]:
        delay = backoff
        if jitter:
            delay += random.uniform(0, backoff * constants.BACKOFF_JITTER_RATIO)
        if max_wait is not None:
            delay = min(delay, max_wait)
        next_backoff = min(
            backoff * constants.BACKOFF_MULTIPLIER, constants.BACKOFF_MAX
        )
        return delay, next_backoff

    def strip_internal_fields(
        self, events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            {k: v for k, v in event.items() if not k.startswith("__we_")}
            for event in events
        ]

    def notify_delivery_failure(self, reason: str, dropped_count: int) -> None:
        if self.on_delivery_failure is None:
            return
        try:
            self.on_delivery_failure(reason, dropped_count, self.queue.length())
        except (
            Exception
        ) as exc:  # pragma: no cover - user callback failures are non-fatal
            logger.warning("wildedge: on_delivery_failure callback failed: %s", exc)

    def dead_letter_and_drop(
        self,
        *,
        reason: str,
        events: list[dict[str, Any]],
        batch_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if self.dead_letter_store is not None:
            self.dead_letter_store.write(
                reason=reason,
                events=self.strip_internal_fields(events),
                batch_id=batch_id,
                details=details,
            )
        self.queue.remove_first_n(len(events))
        self.notify_delivery_failure(reason, len(events))

    def drain_once(self) -> bool:
        events = self.queue.peek_many(self.batch_size)
        if not events:
            return False

        now_unix = time.time()
        expired_count = 0
        for event in events:
            first_seen = float(event.get("__we_first_queued_at", now_unix))
            if (now_unix - first_seen) > self.max_event_age_sec:
                expired_count += 1
            else:
                break
        if expired_count > 0:
            expired = events[:expired_count]
            self.dead_letter_and_drop(
                reason="event_age_exceeded",
                events=expired,
                details={"max_event_age_sec": self.max_event_age_sec},
            )
            logger.warning(
                "wildedge: dropped %d stale queued events (age > %.1fs)",
                expired_count,
                self.max_event_age_sec,
            )
            return True

        for event in events:
            event["__we_attempts"] = int(event.get("__we_attempts", 0)) + 1

        batch = build_batch(
            device=self.device,
            models=self.get_models(),
            events=events,
            session_id=self.session_id,
            created_at=self.created_at,
        )

        if self.debug:
            logger.debug(
                "wildedge: transmitting %d events (batch_id=%s)",
                len(events),
                batch["batch_id"],
            )

        try:
            response = self.transmitter.send(batch)
        except TransmitError as exc:
            logger.warning("wildedge: transmit failed, will retry: %s", exc)
            return False

        if response.status in ("accepted", "partial"):
            self.queue.remove_first_n(len(events))
            if self.debug:
                logger.debug(
                    "wildedge: accepted=%d rejected=%d",
                    response.events_accepted,
                    response.events_rejected,
                )
            return True

        # Permanent client/config errors should not be retried forever.
        # Transmitter returns status="error" for non-retryable cases like 3xx/404.
        if response.status in ("rejected", "unauthorized", "error"):
            self.dead_letter_and_drop(
                reason=f"permanent_{response.status}",
                events=events,
                batch_id=batch["batch_id"],
                details={
                    "response_status": response.status,
                    "events_accepted": response.events_accepted,
                    "events_rejected": response.events_rejected,
                },
            )
            return True

        return False

    def flush(self, timeout: float = 5.0) -> None:
        """Block until the queue drains or timeout expires."""
        if self.stopped:
            return
        deadline = time.monotonic() + timeout
        backoff = constants.BACKOFF_MIN
        while self.queue.length() > 0 and time.monotonic() < deadline:
            progressed = self.drain_once()
            if self.queue.length() == 0:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if progressed:
                backoff = constants.BACKOFF_MIN
                continue
            sleep_for, backoff = self.next_retry_delay(
                backoff,
                jitter=True,
                max_wait=remaining,
            )
            time.sleep(sleep_for)

    def stop(self) -> None:
        """Signal the consumer to stop and wait for thread exit."""
        self.stopped = True
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def close(self, timeout: float | None = None) -> None:
        if timeout is None:
            timeout = constants.DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC
        self.flush(timeout=timeout)
        self.stop()
        self.transmitter.close()
