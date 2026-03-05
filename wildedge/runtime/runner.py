"""Child process runner used by `wildedge run`."""

from __future__ import annotations

import argparse
import os
import runpy
import sys

from wildedge.runtime.bootstrap import (
    RUN_PRINT_STARTUP_REPORT_ENV,
    RUN_PROPAGATE_ENV,
    RuntimeConfigError,
    RuntimeStrictIntegrationError,
    _as_bool,
    clear_runtime_env,
    format_startup_report,
    install_runtime,
)

EXIT_CONFIG_ERROR = 120
EXIT_STRICT_INTEGRATION_ERROR = 121
EXIT_BOOTSTRAP_INTERNAL_ERROR = 122


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m wildedge.runtime.runner")
    parser.add_argument("--mode", choices=["script", "module"], required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(argv)
    args = parsed.args
    if args and args[0] == "--":
        args = args[1:]

    try:
        context = install_runtime()
    except RuntimeConfigError as exc:
        print(f"wildedge: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except RuntimeStrictIntegrationError as exc:
        print(f"wildedge: {exc}", file=sys.stderr)
        return EXIT_STRICT_INTEGRATION_ERROR
    except Exception as exc:
        print(f"wildedge: bootstrap internal error: {exc}", file=sys.stderr)
        return EXIT_BOOTSTRAP_INTERNAL_ERROR

    if (
        getattr(context, "debug", False)
        or getattr(context, "print_startup_report", False)
        or _as_bool(os.environ.get(RUN_PRINT_STARTUP_REPORT_ENV))
    ):
        print(format_startup_report(context), file=sys.stderr)

    if not _as_bool(os.environ.get(RUN_PROPAGATE_ENV, "1")):
        clear_runtime_env()
    try:
        if parsed.mode == "script":
            sys.argv = [parsed.target, *args]
            runpy.run_path(parsed.target, run_name="__main__")
        else:
            sys.argv = [parsed.target, *args]
            runpy.run_module(parsed.target, run_name="__main__", alter_sys=True)
    finally:
        context.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
