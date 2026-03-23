"""WildEdge autoload sitecustomize.

Loaded automatically by the Python interpreter when wildedge/autoload/ is
prepended to PYTHONPATH (by `wildedge run`). Calls install_runtime() before
any user code runs, enabling framework instrumentation and fork-safe operation
for gunicorn, celery, and other pre-fork servers.
"""

from __future__ import annotations

import os
import sys

_AUTOLOAD = "WILDEDGE_AUTOLOAD"  # set to "1" by `wildedge run`
_DSN = "WILDEDGE_DSN"  # user-configured DSN

# sys.modules is used as the guard against re-entry within the same interpreter.
# A module-level variable resets when the module object is re-created (e.g. if
# _load_existing_sitecustomize finds this file via a second sys.path entry in an
# editable install). os.environ would persist but propagates to child processes
# spawned via exec (e.g. uvicorn --reload workers) and blocks their bootstrap.
# sys.modules is process-local and survives module re-execution.
_INSTALLED_MARKER = "wildedge.__autoload_installed__"

if _INSTALLED_MARKER not in sys.modules:
    # Activate if the CLI launched this process, or if WILDEDGE_DSN is set
    # and the user has manually prepended wildedge/autoload/ to PYTHONPATH.
    if os.environ.get(_AUTOLOAD) or os.environ.get(_DSN):
        # Set the guard before importing wildedge to prevent re-entry.
        sys.modules[_INSTALLED_MARKER] = True  # type: ignore[assignment]
        try:
            from wildedge.runtime.bootstrap import install_runtime  # noqa: PLC0415

            # Don't install signal handlers: the host process (gunicorn, celery,
            # etc.) manages SIGTERM/SIGINT itself.
            install_runtime(install_signal_handlers=False)
        except Exception as exc:  # pragma: no cover
            print(f"wildedge: bootstrap failed: {exc}", file=sys.stderr)


# Chain any pre-existing sitecustomize that would otherwise be shadowed.
# Use importlib to find and exec it directly; avoids sys.modules manipulation
# which can trigger CPython's module GC and clear globals mid-execution.
def _load_existing_sitecustomize() -> None:
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


_load_existing_sitecustomize()
