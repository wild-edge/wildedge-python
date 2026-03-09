from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from wildedge import constants

RUN_DSN_ENV = "WILDEDGE_RUN_DSN"
RUN_APP_VERSION_ENV = "WILDEDGE_RUN_APP_VERSION"
RUN_DEBUG_ENV = "WILDEDGE_RUN_DEBUG"
RUN_FLUSH_TIMEOUT_ENV = "WILDEDGE_RUN_FLUSH_TIMEOUT"
RUN_INTEGRATIONS_ENV = "WILDEDGE_RUN_INTEGRATIONS"
RUN_HUBS_ENV = "WILDEDGE_RUN_HUBS"
RUN_STRICT_INTEGRATIONS_ENV = "WILDEDGE_RUN_STRICT_INTEGRATIONS"
RUN_PROPAGATE_ENV = "WILDEDGE_RUN_PROPAGATE"
RUN_PRINT_STARTUP_REPORT_ENV = "WILDEDGE_RUN_PRINT_STARTUP_REPORT"

TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ClientEnv:
    dsn: str | None
    debug: bool
    app_identity: str | None


@dataclass(frozen=True)
class RuntimeEnv:
    dsn: str | None
    app_version: str | None
    debug: bool
    print_startup_report: bool
    strict_integrations: bool
    flush_timeout: float
    integrations: list[str]
    hubs: list[str]
    propagate: bool


@dataclass(frozen=True)
class RunnerEnv:
    print_startup_report: bool
    propagate: bool


def parse_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUE_VALUES


def parse_integration_list(value: str | None, all_values: list[str]) -> list[str]:
    """Parse ``--integrations`` value; defaults to all when unset."""
    if not value or value == "all":
        return sorted(all_values)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_hub_list(value: str | None, all_values: list[str]) -> list[str]:
    """Parse ``--hubs`` value; defaults to empty when unset."""
    if not value or value == "none":
        return []
    if value == "all":
        return sorted(all_values)
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_app_identity(
    *,
    explicit: str | None,
    project_key: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = environ if environ is not None else os.environ
    return explicit or env.get(constants.ENV_APP_IDENTITY) or project_key


def read_client_env(
    *,
    dsn: str | None = None,
    debug: bool | None = None,
    app_identity: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ClientEnv:
    env = environ if environ is not None else os.environ
    resolved_dsn = dsn or env.get(constants.ENV_DSN)
    resolved_debug = (
        debug if debug is not None else parse_bool(env.get(constants.ENV_DEBUG))
    )
    resolved_identity = app_identity or env.get(constants.ENV_APP_IDENTITY)
    return ClientEnv(
        dsn=resolved_dsn,
        debug=resolved_debug,
        app_identity=resolved_identity,
    )


def read_runtime_env(
    *,
    all_integrations: list[str],
    all_hubs: list[str],
    environ: Mapping[str, str] | None = None,
) -> RuntimeEnv:
    env = environ if environ is not None else os.environ
    flush_timeout = float(
        env.get(
            RUN_FLUSH_TIMEOUT_ENV,
            str(constants.DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC),
        )
    )
    return RuntimeEnv(
        dsn=env.get(RUN_DSN_ENV) or env.get(constants.ENV_DSN),
        app_version=env.get(RUN_APP_VERSION_ENV),
        debug=parse_bool(env.get(RUN_DEBUG_ENV)),
        print_startup_report=parse_bool(env.get(RUN_PRINT_STARTUP_REPORT_ENV)),
        strict_integrations=parse_bool(env.get(RUN_STRICT_INTEGRATIONS_ENV)),
        flush_timeout=flush_timeout,
        integrations=parse_integration_list(
            env.get(RUN_INTEGRATIONS_ENV), all_integrations
        ),
        hubs=parse_hub_list(env.get(RUN_HUBS_ENV), all_hubs),
        propagate=parse_bool(env.get(RUN_PROPAGATE_ENV, "1")),
    )


def read_runner_env(
    *,
    environ: Mapping[str, str] | None = None,
) -> RunnerEnv:
    env = environ if environ is not None else os.environ
    return RunnerEnv(
        print_startup_report=parse_bool(env.get(RUN_PRINT_STARTUP_REPORT_ENV)),
        propagate=parse_bool(env.get(RUN_PROPAGATE_ENV, "1")),
    )
