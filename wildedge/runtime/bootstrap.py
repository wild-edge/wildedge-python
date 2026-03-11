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

from wildedge import constants
from wildedge.client import WildEdge
from wildedge.constants import ENV_DSN, WILDEDGE_AUTOLOAD
from wildedge.hubs.registry import HUBS_BY_NAME, supported_hubs
from wildedge.integrations.registry import INTEGRATIONS_BY_NAME, supported_integrations
from wildedge.settings import read_runtime_env

SUPPORTED_SIGNALS = [signal.SIGINT, signal.SIGTERM]
STATUS_OK_PATCHED = "OK_PATCHED"
STATUS_OK_NOOP = "OK_NOOP"
STATUS_SKIP_MISSING_DEP = "SKIP_MISSING_DEP"
STATUS_ERROR_PATCH_FAILED = "ERROR_PATCH_FAILED"
STRICT_FAILURE_STATUSES = {STATUS_SKIP_MISSING_DEP, STATUS_ERROR_PATCH_FAILED}


def clear_runtime_env() -> None:
    """Remove run-scoped env vars so nested processes do not inherit runtime config."""
    for key in (
        WILDEDGE_AUTOLOAD,
        constants.ENV_APP_VERSION,
        constants.ENV_DEBUG,
        constants.ENV_FLUSH_TIMEOUT,
        constants.ENV_INTEGRATIONS,
        constants.ENV_HUBS,
        constants.ENV_STRICT_INTEGRATIONS,
        constants.ENV_PROPAGATE,
        constants.ENV_PRINT_STARTUP_REPORT,
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


def install_runtime(*, install_signal_handlers: bool = True) -> RuntimeContext:
    """Create and configure WildEdge client for process-level instrumentation."""
    try:
        env = read_runtime_env(
            all_integrations=sorted(supported_integrations()),
            all_hubs=sorted(supported_hubs()),
        )
    except ValueError as exc:
        raise RuntimeConfigError("invalid flush timeout") from exc

    if not env.dsn:
        raise RuntimeConfigError(f"{ENV_DSN} must be set to use `wildedge run`.")

    client = WildEdge(dsn=env.dsn, app_version=env.app_version, debug=env.debug)
    statuses: list[dict[str, str]] = []

    for integration in env.integrations:
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
            m for m in spec.required_modules if importlib.util.find_spec(m) is None
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
            if env.debug:
                print(
                    f"wildedge: instrument({integration!r}) failed: {exc}",
                    file=os.sys.stderr,
                )

    for hub_name in env.hubs:
        spec = HUBS_BY_NAME.get(hub_name)
        if spec is None:
            statuses.append(
                {
                    "name": hub_name,
                    "status": STATUS_ERROR_PATCH_FAILED,
                    "detail": "unknown hub",
                }
            )
            continue
        missing = [
            m for m in spec.required_modules if importlib.util.find_spec(m) is None
        ]
        if missing:
            statuses.append(
                {
                    "name": hub_name,
                    "status": STATUS_SKIP_MISSING_DEP,
                    "detail": f"missing modules: {', '.join(missing)}",
                }
            )
            continue
        try:
            client.instrument(None, hubs=[hub_name])
            statuses.append(
                {"name": hub_name, "status": STATUS_OK_PATCHED, "detail": ""}
            )
        except Exception as exc:
            statuses.append(
                {
                    "name": hub_name,
                    "status": STATUS_ERROR_PATCH_FAILED,
                    "detail": str(exc),
                }
            )
            if env.debug:
                print(
                    f"wildedge: instrument(None, hubs=[{hub_name!r}]) failed: {exc}",
                    file=os.sys.stderr,
                )

    if env.strict_integrations:
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
        flush_timeout=env.flush_timeout,
        debug=env.debug,
        print_startup_report=env.print_startup_report,
        integration_statuses=statuses,
    )
    atexit.register(context.shutdown)

    if install_signal_handlers:

        def _handle(sig_num, _frame):  # type: ignore[no-untyped-def]
            context.shutdown()
            raise SystemExit(128 + sig_num)

        for sig in SUPPORTED_SIGNALS:
            signal.signal(sig, _handle)

    return context
