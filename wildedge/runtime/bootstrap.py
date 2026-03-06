"""Process bootstrap for `wildedge run` child execution."""

from __future__ import annotations

import atexit
import importlib.util
import os
import platform
import signal
import threading
from dataclasses import dataclass, field
from importlib import metadata

from wildedge.client import WildEdge
from wildedge.config import DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC, ENV_DSN
from wildedge.integrations.registry import INTEGRATIONS_BY_NAME, supported_integrations

RUN_DSN_ENV = "WILDEDGE_RUN_DSN"
RUN_APP_VERSION_ENV = "WILDEDGE_RUN_APP_VERSION"
RUN_DEBUG_ENV = "WILDEDGE_RUN_DEBUG"
RUN_FLUSH_TIMEOUT_ENV = "WILDEDGE_RUN_FLUSH_TIMEOUT"
RUN_INTEGRATIONS_ENV = "WILDEDGE_RUN_INTEGRATIONS"
RUN_STRICT_INTEGRATIONS_ENV = "WILDEDGE_RUN_STRICT_INTEGRATIONS"
RUN_PROPAGATE_ENV = "WILDEDGE_RUN_PROPAGATE"
RUN_PRINT_STARTUP_REPORT_ENV = "WILDEDGE_RUN_PRINT_STARTUP_REPORT"

SUPPORTED_SIGNALS = [signal.SIGINT, signal.SIGTERM]
STATUS_OK_PATCHED = "OK_PATCHED"
STATUS_OK_NOOP = "OK_NOOP"
STATUS_SKIP_MISSING_DEP = "SKIP_MISSING_DEP"
STATUS_ERROR_PATCH_FAILED = "ERROR_PATCH_FAILED"
STRICT_FAILURE_STATUSES = {STATUS_SKIP_MISSING_DEP, STATUS_ERROR_PATCH_FAILED}


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
        RUN_PRINT_STARTUP_REPORT_ENV,
    ):
        os.environ.pop(key, None)


@dataclass
class RuntimeContext:
    """Holds runtime client and provides idempotent shutdown."""

    client: WildEdge
    flush_timeout: float
    debug: bool
    print_startup_report: bool
    integration_statuses: list[dict[str, str]]
    _closed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.client.flush(timeout=self.flush_timeout)
        self.client.close()


class RuntimeConfigError(RuntimeError):
    """Raised for invalid runtime configuration."""


class RuntimeStrictIntegrationError(RuntimeError):
    """Raised when strict integration mode encounters a non-OK integration status."""


def _sdk_version() -> str:
    try:
        return metadata.version("wildedge-sdk")
    except metadata.PackageNotFoundError:
        return "unknown"


def format_startup_report(context: RuntimeContext) -> str:
    """Render startup diagnostics report."""
    lines = [
        "wildedge startup report",
        f"sdk_version: {_sdk_version()}",
        f"python: {platform.python_version()}",
        f"platform: {platform.platform()}",
        "integrations:",
    ]
    for entry in context.integration_statuses:
        detail = f" ({entry['detail']})" if entry["detail"] else ""
        lines.append(f"- {entry['name']}: {entry['status']}{detail}")
    return "\n".join(lines)


def install_runtime() -> RuntimeContext:
    """Create and configure WildEdge client for process-level instrumentation."""
    dsn = os.environ.get(RUN_DSN_ENV) or os.environ.get(ENV_DSN)
    if not dsn:
        raise RuntimeConfigError(
            f"{ENV_DSN} (or {RUN_DSN_ENV}) must be set to use `wildedge run`."
        )

    app_version = os.environ.get(RUN_APP_VERSION_ENV)
    debug = _as_bool(os.environ.get(RUN_DEBUG_ENV))
    print_startup_report = _as_bool(os.environ.get(RUN_PRINT_STARTUP_REPORT_ENV))
    strict_integrations = _as_bool(os.environ.get(RUN_STRICT_INTEGRATIONS_ENV))
    try:
        flush_timeout = float(
            os.environ.get(
                RUN_FLUSH_TIMEOUT_ENV,
                str(DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC),
            )
        )
    except ValueError as exc:
        raise RuntimeConfigError("invalid flush timeout") from exc

    client = WildEdge(dsn=dsn, app_version=app_version, debug=debug)
    integrations = _integration_list(os.environ.get(RUN_INTEGRATIONS_ENV))
    statuses: list[dict[str, str]] = []
    for integration in integrations:
        spec = INTEGRATIONS_BY_NAME.get(integration)
        if spec is None:
            statuses.append(
                {
                    "name": integration,
                    "status": STATUS_ERROR_PATCH_FAILED,
                    "detail": "unknown integration",
                }
            )
            continue
        missing = [
            module
            for module in spec.required_modules
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            statuses.append(
                {
                    "name": integration,
                    "status": STATUS_SKIP_MISSING_DEP,
                    "detail": f"missing modules: {', '.join(missing)}",
                }
            )
            continue
        try:
            client.instrument(integration)
            status = STATUS_OK_NOOP if spec.kind == "noop" else STATUS_OK_PATCHED
            statuses.append({"name": integration, "status": status, "detail": ""})
        except Exception as exc:
            statuses.append(
                {
                    "name": integration,
                    "status": STATUS_ERROR_PATCH_FAILED,
                    "detail": str(exc),
                }
            )
            if debug:
                print(
                    f"wildedge: instrument({integration!r}) failed: {exc}",
                    file=os.sys.stderr,
                )

    if strict_integrations:
        failures = [row for row in statuses if row["status"] in STRICT_FAILURE_STATUSES]
        if failures:
            fail_detail = ", ".join(
                f"{row['name']}={row['status']}" for row in failures
            )
            raise RuntimeStrictIntegrationError(
                f"strict integration mode failed: {fail_detail}"
            )

    context = RuntimeContext(
        client=client,
        flush_timeout=flush_timeout,
        debug=debug,
        print_startup_report=print_startup_report,
        integration_statuses=statuses,
    )
    atexit.register(context.shutdown)

    def _handle(sig_num, _frame):  # type: ignore[no-untyped-def]
        context.shutdown()
        raise SystemExit(128 + sig_num)

    for sig in SUPPORTED_SIGNALS:
        signal.signal(sig, _handle)

    return context
