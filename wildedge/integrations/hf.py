"""HuggingFace Hub integration that intercepts hf_hub_download / snapshot_download."""

from __future__ import annotations

import os
import sys
import threading
import time

from wildedge.logging import logger
from wildedge.timing import elapsed_ms

try:
    import huggingface_hub as _hf
    import huggingface_hub.file_download as _fd
except ImportError:
    _hf = None  # type: ignore[assignment]
    _fd = None  # type: ignore[assignment]

_local = threading.local()
_hf_hub_download_patched = False
_snapshot_download_patched = False


def _buffer() -> list[dict]:
    if not hasattr(_local, "records"):
        _local.records = []
    return _local.records


def drain_downloads() -> list[dict]:
    """Return and clear buffered download records for the current thread."""
    records = _buffer()
    result = list(records)
    records.clear()
    return result


def _record_download(
    *,
    repo_id: str,
    size: int,
    duration_ms: int,
    cache_hit: bool,
    bandwidth_bps: int | None,
) -> None:
    _buffer().append(
        {
            "repo_id": repo_id,
            "size": size,
            "duration_ms": duration_ms,
            "cache_hit": cache_hit,
            "bandwidth_bps": bandwidth_bps,
            "source_type": "huggingface",
            "source_url": f"hf://{repo_id}",
        }
    )


def _patch_symbol_references(symbol: str, original: object, replacement: object) -> int:
    patched_count = 0
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, symbol, None) is original:
                setattr(mod, symbol, replacement)
                patched_count += 1
        except Exception as exc:
            module_name = getattr(mod, "__name__", "<unknown>")
            logger.debug(
                "wildedge: failed to patch %s in module %s: %s",
                symbol,
                module_name,
                exc,
            )
            continue
    return patched_count


def _install_hf_hub_download_patch() -> bool:
    try:
        orig_hf = _fd.hf_hub_download  # type: ignore[union-attr]
    except Exception:
        logger.exception(
            "wildedge: failed to read huggingface_hub.file_download.hf_hub_download"
        )
        return False

    def tracked_hf_hub_download(repo_id: str, filename: str, **kwargs: object) -> str:
        t0 = time.perf_counter()
        path: str = orig_hf(repo_id, filename, **kwargs)  # type: ignore[arg-type]
        ms = elapsed_ms(t0)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        cache_hit = ms < 200
        bps = int(size * 8 / (ms / 1000)) if ms > 0 and not cache_hit else None
        logger.debug(
            "wildedge: hf_hub_download repo=%s file=%s cache_hit=%s",
            repo_id,
            filename,
            cache_hit,
        )
        _record_download(
            repo_id=repo_id,
            size=size,
            duration_ms=ms,
            cache_hit=cache_hit,
            bandwidth_bps=bps,
        )
        return path

    patched_count = _patch_symbol_references(
        "hf_hub_download", orig_hf, tracked_hf_hub_download
    )
    logger.debug("wildedge: hf_hub_download patched in %d namespace(s)", patched_count)
    return True


def _install_snapshot_download_patch() -> bool:
    snap_orig = getattr(_hf, "snapshot_download", None)
    if snap_orig is None:
        logger.debug("wildedge: huggingface_hub.snapshot_download not available")
        return True

    def tracked_snapshot_download(repo_id: str, **kwargs: object) -> str:
        t0 = time.perf_counter()
        path: str = snap_orig(repo_id, **kwargs)  # type: ignore[arg-type]
        ms = elapsed_ms(t0)
        try:
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(path)
                for f in fs
                if not os.path.islink(os.path.join(dp, f))
            )
        except Exception as exc:
            logger.debug(
                "wildedge: snapshot_download size calculation failed for %s: %s",
                repo_id,
                exc,
            )
            size = 0
        cache_hit = ms < 1000
        bps = int(size * 8 / (ms / 1000)) if ms > 0 and not cache_hit else None
        logger.debug(
            "wildedge: snapshot_download repo=%s cache_hit=%s",
            repo_id,
            cache_hit,
        )
        _record_download(
            repo_id=repo_id,
            size=size,
            duration_ms=ms,
            cache_hit=cache_hit,
            bandwidth_bps=bps,
        )
        return path

    patched_count = _patch_symbol_references(
        "snapshot_download", snap_orig, tracked_snapshot_download
    )
    logger.debug(
        "wildedge: snapshot_download patched in %d namespace(s)", patched_count
    )
    return True


def install_patch() -> None:
    global _hf_hub_download_patched, _snapshot_download_patched

    if _hf_hub_download_patched and _snapshot_download_patched:
        return

    if _hf is None or _fd is None:
        logger.warning(
            "wildedge: huggingface_hub not installed; instrument('huggingface') skipped"
        )
        return

    if not _hf_hub_download_patched:
        _hf_hub_download_patched = _install_hf_hub_download_patch()
    if not _snapshot_download_patched:
        _snapshot_download_patched = _install_snapshot_download_patch()
