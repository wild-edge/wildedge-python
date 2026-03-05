"""Process bootstrap for `wildedge run` child execution."""

from __future__ import annotations

import atexit
import os
import signal
import threading
from dataclasses import dataclass, field

from wildedge.client import WildEdge
from wildedge.config import ENV_DSN
from wildedge.integrations.registry import supported_integrations

RUN_DSN_ENV = "WILDEDGE_RUN_DSN"
RUN_APP_VERSION_ENV = "WILDEDGE_RUN_APP_VERSION"
RUN_DEBUG_ENV = "WILDEDGE_RUN_DEBUG"
RUN_FLUSH_TIMEOUT_ENV = "WILDEDGE_RUN_FLUSH_TIMEOUT"
RUN_INTEGRATIONS_ENV = "WILDEDGE_RUN_INTEGRATIONS"
RUN_STRICT_INTEGRATIONS_ENV = "WILDEDGE_RUN_STRICT_INTEGRATIONS"
RUN_PROPAGATE_ENV = "WILDEDGE_RUN_PROPAGATE"

SUPPORTED_SIGNALS = [signal.SIGINT, signal.SIGTERM]


def _as_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _integration_list(value: str | None) -> list[str]:
    if not value or value == "all":
        return sorted(supported_integrations())
    return [item.strip() for item in value.split(",") if item.strip()]


def clear_runtime_env() -> None:
    """Remove run-scoped env vars so nested processes do not inherit runtime config."""
    for key in (
        RUN_DSN_ENV,
        RUN_APP_VERSION_ENV,
        RUN_DEBUG_ENV,
        RUN_FLUSH_TIMEOUT_ENV,
        RUN_INTEGRATIONS_ENV,
        RUN_STRICT_INTEGRATIONS_ENV,
        RUN_PROPAGATE_ENV,
    ):
        os.environ.pop(key, None)


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
    strict_integrations = _as_bool(os.environ.get(RUN_STRICT_INTEGRATIONS_ENV))
    flush_timeout = float(os.environ.get(RUN_FLUSH_TIMEOUT_ENV, "5.0"))

    client = WildEdge(dsn=dsn, app_version=app_version, debug=debug)
    integrations = _integration_list(os.environ.get(RUN_INTEGRATIONS_ENV))
    for integration in integrations:
        try:
            client.instrument(integration)
        except Exception as exc:
            if strict_integrations:
                raise RuntimeError(
                    f"Failed to instrument integration {integration!r}: {exc}"
                ) from exc
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
