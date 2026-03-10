"""WildEdge autoload sitecustomize.

Loaded automatically by the Python interpreter when wildedge/autoload/ is
prepended to PYTHONPATH (by `wildedge run`). Calls install_runtime() before
any user code runs, enabling framework instrumentation and fork-safe operation
for gunicorn, celery, and other pre-fork servers.
"""

from __future__ import annotations

import os
import sys

_GUARD = "WILDEDGE_AUTOLOAD_ACTIVE"
_RUN_DSN = "WILDEDGE_RUN_DSN"
_DSN = "WILDEDGE_DSN"


def _bootstrap() -> None:
    # Idempotent: skip if already initialized (e.g. in a forked worker that
    # exec'd a subprocess, or if sitecustomize.py is imported twice).
    if os.environ.get(_GUARD):
        return

    # Require a DSN. Silently skip if missing so that processes which
    # inherit PYTHONPATH without wildedge config don't crash.
    if not (os.environ.get(_RUN_DSN) or os.environ.get(_DSN)):
        return

    # Set the guard before importing wildedge to prevent re-entry.
    os.environ[_GUARD] = "1"

    try:
        from wildedge.runtime.bootstrap import install_runtime  # noqa: PLC0415

        # Don't install signal handlers: the host process (gunicorn, celery,
        # etc.) manages SIGTERM/SIGINT itself.
        install_runtime(install_signal_handlers=False)
    except Exception as exc:  # pragma: no cover
        print(f"wildedge: bootstrap failed: {exc}", file=sys.stderr)


_bootstrap()


# Chain any pre-existing sitecustomize that would otherwise be shadowed.
# Use importlib to find and exec it directly — avoids sys.modules manipulation
# which can trigger CPython's module GC and clear globals mid-execution.
def _chain_sitecustomize() -> None:
    import importlib.util as _iutil

    _autoload_dir = os.path.dirname(os.path.abspath(__file__))
    _saved = sys.path[:]
    sys.path = [p for p in sys.path if p != _autoload_dir]
    try:
        _spec = _iutil.find_spec("sitecustomize")
        if _spec is not None and _spec.origin is not None:
            _mod = _iutil.module_from_spec(_spec)
            sys.modules["sitecustomize"] = _mod
            _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    except Exception:
        pass
    finally:
        sys.path = _saved


_chain_sitecustomize()
