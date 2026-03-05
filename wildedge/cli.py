"""Command-line entry point for WildEdge."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

from wildedge.client import parse_dsn
from wildedge.integrations.registry import INTEGRATIONS_BY_NAME, supported_integrations
from wildedge.runtime.bootstrap import (
    RUN_APP_VERSION_ENV,
    RUN_DEBUG_ENV,
    RUN_DSN_ENV,
    RUN_FLUSH_TIMEOUT_ENV,
    RUN_INTEGRATIONS_ENV,
    RUN_PROPAGATE_ENV,
    RUN_STRICT_INTEGRATIONS_ENV,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wildedge")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a Python program with WildEdge runtime enabled.")
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
    doctor.add_argument("--dsn", default=None, help="DSN to validate (or use WILDEDGE_DSN).")
    doctor.add_argument(
        "--integrations",
        default="all",
        help="Comma-separated integrations to validate imports for. Default: all.",
    )
    return parser


def _is_python_executable(token: str) -> bool:
    name = Path(token).name.lower()
    return name.startswith("python")


def _parse_python_command(tokens: list[str]) -> tuple[str, str, str, list[str]]:
    if not tokens:
        raise ValueError("missing command after `wildedge run --`")

    args = list(tokens)
    if args[0] == "--":
        args = args[1:]
    if not args:
        raise ValueError("missing python command after `--`")

    python_exe = sys.executable
    if _is_python_executable(args[0]):
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


def _run_command(parsed: argparse.Namespace) -> int:
    try:
        python_exe, mode, target, args = _parse_python_command(parsed.command_args)
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


def _integration_list(value: str | None) -> list[str]:
    if not value or value == "all":
        return sorted(supported_integrations())
    return [item.strip() for item in value.split(",") if item.strip()]


def _doctor(parsed: argparse.Namespace) -> int:
    ok = True
    dsn = parsed.dsn or os.environ.get("WILDEDGE_DSN")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")

    if not dsn:
        ok = False
        print("dsn: FAIL (missing WILDEDGE_DSN or --dsn)")
    else:
        try:
            parse_dsn(dsn)
            print("dsn: OK")
        except Exception as exc:
            ok = False
            print(f"dsn: FAIL ({exc})")

    for integration in _integration_list(parsed.integrations):
        spec = INTEGRATIONS_BY_NAME.get(integration)
        if spec is None:
            ok = False
            print(f"integration[{integration}]: FAIL (unknown integration)")
            continue

        missing = [
            module for module in spec.required_modules if importlib.util.find_spec(module) is None
        ]
        if missing:
            ok = False
            print(
                f"integration[{integration}]: FAIL "
                f"(missing modules: {', '.join(missing)})"
            )
        else:
            print(f"integration[{integration}]: OK")

    if ok:
        print("doctor: PASS")
        return 0
    print("doctor: FAIL")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(argv)
    if parsed.command == "run":
        return _run_command(parsed)
    if parsed.command == "doctor":
        return _doctor(parsed)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
