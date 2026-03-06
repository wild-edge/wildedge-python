"""Abstract base for model hub/repository trackers."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class BaseHubTracker(ABC):
    """
    Tracks model downloads from a specific model hub or repository.

    Hub trackers handle two complementary use cases:

    **Thread-local interception** (``install_patch`` / ``drain``)
        Used when the hub library's download functions are patched directly.
        Records accumulate in a per-thread buffer and are drained by the client
        when a model load event is emitted (e.g. ONNX auto-load, explicit
        ``hf_hub_download`` calls).

    **Filesystem diff** (``scan_cache`` / ``diff_to_records``)
        Used by framework integrations (e.g. timm) where downloads happen
        implicitly inside a model construction call and there is no single
        download function to intercept. The integration snapshots the cache
        before and after ``create_model()`` and derives download records from
        new files that appeared.

    Both mechanisms are complementary. The client's ``_on_model_auto_loaded``
    prefers caller-supplied filesystem-diff records and drains thread-local
    buffers only when no diff records are available (or to keep buffers clean).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tracker identifier, e.g. ``'huggingface'`` or ``'torchhub'``."""

    @abstractmethod
    def can_install(self) -> bool:
        """Return True if the required library is importable."""

    @abstractmethod
    def install_patch(self, client_ref: object) -> None:
        """
        Monkey-patch the hub library's download or load functions.

        Called at most once per process. ``client_ref`` is a ``weakref.ref``
        to the ``WildEdge`` client; dereference it before use and guard against
        ``None`` (client may have been closed or garbage-collected).
        """

    @abstractmethod
    def drain(self) -> list[dict]:
        """Return and clear buffered download records for the current thread."""

    @abstractmethod
    def cache_dir(self) -> str | None:
        """Return the hub's local cache directory path, or ``None`` if unavailable."""

    def scan_cache(self) -> dict[str, int]:
        """
        Snapshot all real (non-symlink) files in this hub's local cache.

        Returns ``{filepath: size_bytes}``. Symlinks are skipped so HuggingFace
        Hub's ``snapshot/`` entries (which symlink into ``blobs/``) are not
        double-counted.

        Override to customise scanning behaviour for a specific hub layout.
        """
        directory = self.cache_dir()
        if not directory:
            return {}
        snapshot: dict[str, int] = {}
        try:
            for dirpath, _, filenames in os.walk(directory):
                for filename in filenames:
                    path = os.path.join(dirpath, filename)
                    if os.path.islink(path):
                        continue
                    try:
                        snapshot[path] = os.path.getsize(path)
                    except OSError:
                        pass
        except Exception:
            pass
        return snapshot

    @abstractmethod
    def diff_to_records(
        self, before: dict[str, int], after: dict[str, int], elapsed_ms: int
    ) -> list[dict]:
        """
        Convert a before/after cache snapshot diff into download event records.

        Each record must contain:
        ``repo_id``, ``size``, ``duration_ms``, ``cache_hit``,
        ``bandwidth_bps``, ``source_type``, ``source_url``.
        """
