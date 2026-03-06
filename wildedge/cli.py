"""Command-line entry point for WildEdge."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from wildedge import config
from wildedge.client import parse_dsn
from wildedge.device import get_device_id_path
from wildedge.integrations.registry import INTEGRATIONS_BY_NAME, supported_integrations
from wildedge.runtime.bootstrap import (
    RUN_APP_VERSION_ENV,
    RUN_DEBUG_ENV,
    RUN_DSN_ENV,
    RUN_FLUSH_TIMEOUT_ENV,
    RUN_INTEGRATIONS_ENV,
    RUN_PRINT_STARTUP_REPORT_ENV,
    RUN_PROPAGATE_ENV,
    RUN_STRICT_INTEGRATIONS_ENV,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wildedge")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run", help="Run a Python program with WildEdge runtime enabled."
    )
    run.add_argument("--dsn", default=None, help="DSN override for this command.")
    run.add_argument(
        "--app-version",
        default=None,
        help="App version to report in device metadata.",
    )
    run.add_argument(
        "--integrations",
        default="all",
        help="Comma-separated integrations to enable. Default: all.",
    )
    run.add_argument(
        "--flush-timeout",
        type=float,
        default=5.0,
        help="Flush timeout (seconds) for shutdown.",
    )
    run.add_argument(
        "--debug",
        action="store_true",
        help="Enable WildEdge debug logging in child process.",
    )
    run.add_argument(
        "--print-startup-report",
        action="store_true",
        help="Print runtime startup diagnostics before target execution.",
    )
    run.add_argument(
        "--strict-integrations",
        action="store_true",
        help="Fail startup if any requested integration cannot be instrumented.",
    )
    propagation = run.add_mutually_exclusive_group()
    propagation.add_argument(
        "--propagate",
        dest="propagate",
        action="store_true",
        default=True,
        help="Propagate runtime env vars to nested child processes (default).",
    )
    propagation.add_argument(
        "--no-propagate",
        dest="propagate",
        action="store_false",
        help="Do not propagate runtime env vars to nested child processes.",
    )
    run.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
        help="Python command to run, e.g. -- python app.py or -- python -m pkg.module",
    )

    doctor = sub.add_parser("doctor", help="Validate local WildEdge runtime readiness.")
    doctor.add_argument(
        "--dsn", default=None, help="DSN to validate (or use WILDEDGE_DSN)."
    )
    doctor.add_argument(
        "--integrations",
        default="all",
        help="Comma-separated integrations to validate imports for. Default: all.",
    )
    doctor.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    doctor.add_argument(
        "--network-check",
        action="store_true",
        help="Attempt TCP reachability check to DSN host:port.",
    )
    doctor.add_argument(
        "--batch-size",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help="Validate intended batch size against SDK limits.",
    )
    doctor.add_argument(
        "--flush-interval",
        type=float,
        default=config.DEFAULT_FLUSH_INTERVAL_SEC,
        help="Validate intended flush interval against SDK limits.",
    )
    doctor.add_argument(
        "--max-queue-size",
        type=int,
        default=config.DEFAULT_MAX_QUEUE_SIZE,
        help="Validate intended queue size against SDK limits.",
    )
    return parser


def is_python_executable(token: str) -> bool:
    name = Path(token).name.lower()
    return name.startswith("python")


def parse_python_command(tokens: list[str]) -> tuple[str, str, str, list[str]]:
    if not tokens:
        raise ValueError("missing command after `wildedge run --`")

    args = list(tokens)
    if args[0] == "--":
        args = args[1:]
    if not args:
        raise ValueError("missing python command after `--`")

    python_exe = sys.executable
    if is_python_executable(args[0]):
        python_exe = args.pop(0)

    if not args:
        raise ValueError("missing script path or -m <module>")

    if args[0] == "-m":
        if len(args) < 2:
            raise ValueError("`-m` requires a module name")
        return python_exe, "module", args[1], args[2:]

    target = args[0]
    if target.endswith(".py"):
        return python_exe, "script", target, args[1:]

    raise ValueError(
        "unsupported command format; use `python script.py ...` or `python -m module ...`"
    )


def run_command(parsed: argparse.Namespace) -> int:
    try:
        python_exe, mode, target, args = parse_python_command(parsed.command_args)
    except ValueError as exc:
        print(f"wildedge: {exc}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    if parsed.dsn:
        env[RUN_DSN_ENV] = parsed.dsn
    if parsed.app_version:
        env[RUN_APP_VERSION_ENV] = parsed.app_version
    if parsed.debug:
        env[RUN_DEBUG_ENV] = "1"
    env[RUN_INTEGRATIONS_ENV] = parsed.integrations
    env[RUN_FLUSH_TIMEOUT_ENV] = str(parsed.flush_timeout)
    env[RUN_PROPAGATE_ENV] = "1" if parsed.propagate else "0"
    env[RUN_STRICT_INTEGRATIONS_ENV] = "1" if parsed.strict_integrations else "0"
    env[RUN_PRINT_STARTUP_REPORT_ENV] = "1" if parsed.print_startup_report else "0"

    cmd = [
        python_exe,
        "-m",
        "wildedge.runtime.runner",
        "--mode",
        mode,
        "--target",
        target,
        "--",
        *args,
    ]
    completed = subprocess.run(cmd, env=env, check=False)
    return completed.returncode


def integration_list(value: str | None) -> list[str]:
    if not value or value == "all":
        return sorted(supported_integrations())
    return [item.strip() for item in value.split(",") if item.strip()]


def check_writable_dir(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".wildedge_doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, str(path)
    except Exception as exc:
        return False, f"{path} ({exc})"


def validate_runtime_config(parsed: argparse.Namespace) -> tuple[bool, str]:
    if not (config.BATCH_SIZE_MIN <= parsed.batch_size <= config.BATCH_SIZE_MAX):
        return False, "batch_size out of range"
    if not (
        config.FLUSH_INTERVAL_MIN <= parsed.flush_interval <= config.FLUSH_INTERVAL_MAX
    ):
        return False, "flush_interval out of range"
    if not (
        config.MAX_QUEUE_SIZE_MIN <= parsed.max_queue_size <= config.MAX_QUEUE_SIZE_MAX
    ):
        return False, "max_queue_size out of range"
    return True, "OK"


def network_reachability_check(host_url: str) -> tuple[bool, str]:
    parsed = urlparse(host_url)
    host = parsed.hostname
    if not host:
        return False, "missing host"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=2):
            return True, f"{host}:{port}"
    except OSError as exc:
        return False, f"{host}:{port} ({exc})"


def doctor_report(parsed: argparse.Namespace) -> dict:
    report: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "checks": [],
        "integrations": [],
    }
    checks: list[dict[str, str]] = report["checks"]  # type: ignore[assignment]
    integrations: list[dict[str, str]] = report["integrations"]  # type: ignore[assignment]

    ok = True
    dsn = parsed.dsn or os.environ.get("WILDEDGE_DSN")

    if not dsn:
        ok = False
        checks.append(
            {
                "name": "dsn",
                "status": "FAIL",
                "detail": "missing WILDEDGE_DSN or --dsn",
            }
        )
    else:
        try:
            _, host_url = parse_dsn(dsn)
            checks.append({"name": "dsn", "status": "OK", "detail": host_url})
            if parsed.network_check:
                reachable, detail = network_reachability_check(host_url)
                checks.append(
                    {
                        "name": "network",
                        "status": "OK" if reachable else "FAIL",
                        "detail": detail,
                    }
                )
                ok = ok and reachable
        except Exception as exc:
            ok = False
            checks.append({"name": "dsn", "status": "FAIL", "detail": str(exc)})

    temp_ok, temp_detail = check_writable_dir(Path(tempfile.gettempdir()))
    checks.append(
        {
            "name": "writable_tempdir",
            "status": "OK" if temp_ok else "FAIL",
            "detail": temp_detail,
        }
    )
    ok = ok and temp_ok

    config_ok, config_detail = validate_runtime_config(parsed)
    checks.append(
        {
            "name": "runtime_config",
            "status": "OK" if config_ok else "FAIL",
            "detail": config_detail,
        }
    )
    ok = ok and config_ok

    device_dir_ok, device_dir_detail = check_writable_dir(get_device_id_path().parent)
    checks.append(
        {
            "name": "writable_device_config_dir",
            "status": "OK" if device_dir_ok else "FAIL",
            "detail": device_dir_detail,
        }
    )
    ok = ok and device_dir_ok

    for integration in integration_list(parsed.integrations):
        spec = INTEGRATIONS_BY_NAME.get(integration)
        if spec is None:
            ok = False
            integrations.append(
                {
                    "name": integration,
                    "status": "FAIL",
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
            ok = False
            integrations.append(
                {
                    "name": integration,
                    "status": "FAIL",
                    "detail": f"missing modules: {', '.join(missing)}",
                }
            )
        else:
            integrations.append({"name": integration, "status": "OK", "detail": ""})

    report["status"] = "PASS" if ok else "FAIL"
    return report


def print_doctor_text(report: dict) -> None:
    print(f"python: {report['python']}")
    print(f"platform: {report['platform']}")
    for check in report["checks"]:
        name = check["name"]
        status = check["status"]
        detail = check["detail"]
        if detail:
            print(f"{name}: {status} ({detail})")
        else:
            print(f"{name}: {status}")
    for integration in report["integrations"]:
        name = integration["name"]
        status = integration["status"]
        detail = integration["detail"]
        if detail:
            print(f"integration[{name}]: {status} ({detail})")
        else:
            print(f"integration[{name}]: {status}")
    print(f"doctor: {report['status']}")


def doctor(parsed: argparse.Namespace) -> int:
    report = doctor_report(parsed)
    if parsed.format == "json":
        print(json.dumps(report, sort_keys=True))
    else:
        print_doctor_text(report)
    return 0 if report["status"] == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    if parsed.command == "run":
        return run_command(parsed)
    if parsed.command == "doctor":
        return doctor(parsed)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
