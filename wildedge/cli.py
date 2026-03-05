"""Command-line entry point for WildEdge."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from wildedge.runtime.bootstrap import (
    RUN_APP_VERSION_ENV,
    RUN_DEBUG_ENV,
    RUN_DSN_ENV,
    RUN_FLUSH_TIMEOUT_ENV,
    RUN_INTEGRATIONS_ENV,
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
        "command_args",
        nargs=argparse.REMAINDER,
        help="Python command to run, e.g. -- python app.py or -- python -m pkg.module",
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(argv)
    if parsed.command == "run":
        return _run_command(parsed)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

