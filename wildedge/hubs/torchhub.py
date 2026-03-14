"""PyTorch Hub tracker.

Patches ``torch.hub.load`` to time model loads, detect downloaded files via
a before/after filesystem diff of the torch hub cache, and emit a model load
event through the WildEdge client.

Also provides ``scan_cache`` / ``diff_to_records`` so the timm integration can
detect files that land in the torch hub cache directory as a side-effect of
older timm models that use ``torch.hub.download_url_to_file`` for weights
(rather than the HuggingFace Hub path used by modern timm).

Source-type labelling
---------------------
All records produced by this tracker use ``source_type='torchhub'`` and
``source_url='torchhub://<identifier>'``, fixing a bug in the previous
``hf_cache.py`` implementation which labelled these files as
``source_type='url'`` with a local filesystem path as the URL.

Cache layout
------------
``torch.hub.get_dir()`` (default ``~/.cache/torch/hub``) contains:

- ``checkpoints/<name>-<8hexchars>.ext``: weight files downloaded by
  ``torch.hub.download_url_to_file`` or ``torch.utils.model_zoo.load_url``.
- ``<owner>_<repo>_<ref>/``: cloned GitHub repo directories created by
  ``torch.hub.load``.
"""

from __future__ import annotations

import os
import re
import threading
import time

from wildedge.hubs.base import BaseHubTracker
from wildedge.logging import logger
from wildedge.timing import elapsed_ms

try:
    import torch as _torch
except ImportError:
    _torch = None  # type: ignore[assignment]

_local = threading.local()
_torch_hub_load_patched = False
_TORCH_HUB_PATCH_LOCK = threading.Lock()
TORCH_HUB_LOAD_PATCH_NAME = "torchhub_load"

# Matches the hash suffix appended by torch hub to checkpoint filenames,
# e.g. resnet50-0676ba61.pth → strip "-0676ba61" to get "resnet50".
_CHECKPOINT_HASH_RE = re.compile(r"-[0-9a-f]{8,}(?=\.[^.]+$)")


def _buffer() -> list[dict]:
    if not hasattr(_local, "records"):
        _local.records = []
    return _local.records


def _strip_hash_suffix(filename: str) -> str:
    """Return checkpoint filename without the appended content hash."""
    return _CHECKPOINT_HASH_RE.sub("", filename)


def _parse_repo_id(component: str) -> str:
    """
    Parse ``<owner>_<repo>_<ref>`` torch hub directory name into ``owner/repo``.

    torch.hub clones ``github.com/<owner>/<repo>`` and names the local directory
    ``{owner}_{repo}_{ref}`` (e.g. ``pytorch_vision_v0.10.0``).  This is a
    best-effort parse: if the component cannot be split cleanly the original
    string is returned verbatim.
    """
    parts = component.split("_")
    if len(parts) >= 3:
        # Last segment is usually a version/ref tag; owner is first, repo is middle.
        return f"{parts[0]}/{parts[1]}"
    return component


class TorchHubTracker(BaseHubTracker):
    """
    Tracks PyTorch Hub downloads and model loads.

    **Patched entry point**: ``torch.hub.load(repo_or_dir, model, ...)``

    When the patch is active, each ``torch.hub.load`` call:

    1. Snapshots the torch hub cache directory *before* the load.
    2. Calls the original ``torch.hub.load``.
    3. Snapshots the cache *after* to detect newly downloaded files.
    4. Calls ``client._on_model_auto_loaded`` with the resulting download
       records, so load timing and provenance are emitted together.

    **Filesystem diff fallback** (``scan_cache`` / ``diff_to_records``):

    The timm integration uses a before/after cache diff to capture downloads
    that happen implicitly inside ``timm.create_model()``.  Older timm versions
    may download weight files into the torch hub checkpoints directory via
    ``torch.hub.download_url_to_file``.  Including the torch hub cache in the
    timm diff ensures these files are attributed as ``source_type='torchhub'``
    rather than the incorrect ``source_type='url'`` used by the old
    ``hf_cache.py`` implementation.
    """

    @property
    def name(self) -> str:
        return "torchhub"

    def can_install(self) -> bool:
        return _torch is not None

    def cache_dir(self) -> str | None:
        if _torch is None:
            return None
        try:
            return _torch.hub.get_dir()
        except Exception:
            return None

    def drain(self) -> list[dict]:
        records = _buffer()
        result = list(records)
        records.clear()
        return result

    def diff_to_records(
        self, before: dict[str, int], after: dict[str, int], elapsed_ms: int
    ) -> list[dict]:
        """
        Convert a torch hub cache diff into download records.

        Files in ``checkpoints/`` are weight blobs; their hash suffix is stripped
        to produce a human-readable ``repo_id``.  Files in repo clone directories
        (``<owner>_<repo>_<ref>/``) are attributed to ``owner/repo``.
        """
        new_files = {p: s for p, s in after.items() if p not in before}
        if not new_files:
            return []

        hub_dir = self.cache_dir() or ""
        records: list[dict] = []

        for path, size in new_files.items():
            filename = os.path.basename(path)
            try:
                rel = os.path.relpath(path, hub_dir) if hub_dir else path
                parts = rel.split(os.sep)
                if parts[0] == "checkpoints":
                    clean_name = _strip_hash_suffix(filename)
                    repo_id = clean_name
                    source_url = f"torchhub://checkpoints/{filename}"
                else:
                    repo_id = _parse_repo_id(parts[0])
                    source_url = f"torchhub://{repo_id}"
            except Exception:
                repo_id = filename
                source_url = f"torchhub://{filename}"

            logger.debug(
                "wildedge: torchhub cache diff: repo=%s new_bytes=%d", repo_id, size
            )
            bps = int(size * 8 / (elapsed_ms / 1000)) if elapsed_ms > 0 else None
            records.append(
                {
                    "repo_id": repo_id,
                    "size": size,
                    "duration_ms": elapsed_ms,
                    "cache_hit": False,
                    "bandwidth_bps": bps,
                    "source_type": "torchhub",
                    "source_url": source_url,
                }
            )
        return records

    def install_patch(self, client_ref: object) -> None:
        """
        Patch ``torch.hub.load`` for automatic load tracking.

        Each ``torch.hub.load(repo_or_dir, model, ...)`` call is timed; a
        before/after cache diff detects newly downloaded files; the resulting
        download and load events are emitted via
        ``client._on_model_auto_loaded``.  The returned model object is a
        PyTorch module, so the existing ``PytorchExtractor`` handles metadata
        extraction and inference hook installation automatically.
        """
        global _torch_hub_load_patched
        if _torch_hub_load_patched or _torch is None:
            return

        with _TORCH_HUB_PATCH_LOCK:
            if _torch_hub_load_patched:
                return

            original_load = _torch.hub.load
            if (
                getattr(original_load, "__wildedge_patch_name__", None)
                == TORCH_HUB_LOAD_PATCH_NAME
            ):
                _torch_hub_load_patched = True
                return

            tracker = self  # capture for closure

            def patched_load(  # type: ignore[no-untyped-def]
                repo_or_dir, model, *args, **kwargs
            ):
                before = tracker.scan_cache()
                t0 = time.perf_counter()
                result = original_load(repo_or_dir, model, *args, **kwargs)
                load_ms = elapsed_ms(t0)
                after = tracker.scan_cache()
                downloads = tracker.diff_to_records(before, after, load_ms) or None
                c = client_ref()  # type: ignore[call-arg]
                if c is not None and not c.closed:
                    c._on_model_auto_loaded(
                        result, load_ms=load_ms, downloads=downloads
                    )
                return result

            patched_load.__wildedge_patch_name__ = TORCH_HUB_LOAD_PATCH_NAME  # type: ignore[attr-defined]
            patched_load.__wildedge_original_call__ = original_load  # type: ignore[attr-defined]
            _torch.hub.load = patched_load
            _torch_hub_load_patched = True
