"""Process bootstrap for `wildedge run` child execution."""

from __future__ import annotations

import atexit
import os
import signal
import threading
from dataclasses import dataclass, field

from wildedge.client import WildEdge
from wildedge.config import ENV_DSN

RUN_DSN_ENV = "WILDEDGE_RUN_DSN"
RUN_APP_VERSION_ENV = "WILDEDGE_RUN_APP_VERSION"
RUN_DEBUG_ENV = "WILDEDGE_RUN_DEBUG"
RUN_FLUSH_TIMEOUT_ENV = "WILDEDGE_RUN_FLUSH_TIMEOUT"
RUN_INTEGRATIONS_ENV = "WILDEDGE_RUN_INTEGRATIONS"

SUPPORTED_SIGNALS = [signal.SIGINT, signal.SIGTERM]


def _as_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _integration_list(value: str | None) -> list[str]:
    if not value or value == "all":
        return sorted(WildEdge.SUPPORTED_INTEGRATIONS)
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class RuntimeContext:
    """Holds runtime client and provides idempotent shutdown."""

    client: WildEdge
    flush_timeout: float
    debug: bool
    _closed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.client.flush(timeout=self.flush_timeout)
        self.client.close()


def install_runtime() -> RuntimeContext:
    """Create and configure WildEdge client for process-level instrumentation."""
    dsn = os.environ.get(RUN_DSN_ENV) or os.environ.get(ENV_DSN)
    if not dsn:
        raise RuntimeError(
            f"{ENV_DSN} (or {RUN_DSN_ENV}) must be set to use `wildedge run`."
        )

    app_version = os.environ.get(RUN_APP_VERSION_ENV)
    debug = _as_bool(os.environ.get(RUN_DEBUG_ENV))
    flush_timeout = float(os.environ.get(RUN_FLUSH_TIMEOUT_ENV, "5.0"))

    client = WildEdge(dsn=dsn, app_version=app_version, debug=debug)
    integrations = _integration_list(os.environ.get(RUN_INTEGRATIONS_ENV))
    for integration in integrations:
        try:
            client.instrument(integration)
        except Exception as exc:
            if debug:
                print(
                    f"wildedge: instrument({integration!r}) failed: {exc}",
                    file=os.sys.stderr,
                )

    context = RuntimeContext(client=client, flush_timeout=flush_timeout, debug=debug)
    atexit.register(context.shutdown)

    def _handle(sig_num, _frame):  # type: ignore[no-untyped-def]
        context.shutdown()
        raise SystemExit(128 + sig_num)

    for sig in SUPPORTED_SIGNALS:
        signal.signal(sig, _handle)

    return context
