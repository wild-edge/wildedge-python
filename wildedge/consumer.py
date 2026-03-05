from __future__ import annotations

import atexit
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from wildedge import config
from wildedge.batch import build_batch
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
    ):
        self.queue = queue
        self.transmitter = transmitter
        self.device = device
        self.get_models = get_models
        self.session_id = session_id
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        self.debug = debug

        self.stop_event = threading.Event()
        self.stopped = False
        self.backoff = config.BACKOFF_MIN
        self.created_at = datetime.now(timezone.utc)

        self.thread = threading.Thread(
            target=self.run, daemon=True, name="wildedge-consumer"
        )
        self.thread.start()
        atexit.register(self.flush)

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
                    self.backoff = config.BACKOFF_MIN
                else:
                    self.stop_event.wait(timeout=self.backoff)
                    self.backoff = min(
                        self.backoff * config.BACKOFF_MULTIPLIER, config.BACKOFF_MAX
                    )
            else:
                self.stop_event.wait(timeout=config.IDLE_POLL_INTERVAL)

    def drain_once(self) -> bool:
        events = self.queue.peek_many(self.batch_size)
        if not events:
            return False

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

        if response.status in ("rejected", "unauthorized"):
            self.queue.remove_first_n(len(events))
            return True

        return False

    def flush(self, timeout: float = 5.0) -> None:
        """Block until the queue drains or timeout expires."""
        if self.stopped:
            return
        deadline = time.monotonic() + timeout
        while self.queue.length() > 0 and time.monotonic() < deadline:
            self.drain_once()
            if self.queue.length() > 0:
                time.sleep(0.05)

    def stop(self) -> None:
        """Signal the consumer to stop and wait for thread exit."""
        self.stopped = True
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def close(self) -> None:
        self.flush()
        self.stop()
        self.transmitter.close()
