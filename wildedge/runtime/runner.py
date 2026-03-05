"""Child process runner used by `wildedge run`."""

from __future__ import annotations

import argparse
import runpy
import sys

from wildedge.runtime.bootstrap import install_runtime


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

    context = install_runtime()
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

