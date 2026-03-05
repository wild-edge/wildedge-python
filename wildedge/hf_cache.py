"""Shared HF Hub / torch hub cache-diff utilities."""

from __future__ import annotations

import os

from wildedge.logging import logger

try:
    import torch as _torch
except ImportError:
    _torch = None  # type: ignore[assignment]


def scan_model_caches() -> dict[str, int]:
    """Snapshot all real (non-symlink) files in known model cache directories.

    Returns a ``{filepath: size_bytes}`` dict covering the HuggingFace Hub cache
    and the torch hub cache. Symlinks are skipped so HF Hub's snapshot/ entries
    don't double-count blobs/.
    """
    snapshot: dict[str, int] = {}
    hf_dir = os.environ.get(
        "HUGGINGFACE_HUB_CACHE",
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub",
        ),
    )
    try:
        for dirpath, _, filenames in os.walk(hf_dir):
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
    if _torch is not None:
        try:
            torch_dir = _torch.hub.get_dir()
            for dirpath, _, filenames in os.walk(torch_dir):
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


def downloads_from_cache_diff(
    before: dict[str, int], after: dict[str, int], elapsed_ms: int
) -> list[dict]:
    """Return download records for files that appeared in *after* but not *before*.

    HF Hub paths are grouped by repo (``models--org--name`` directory component).
    Torch hub / direct-URL files are emitted individually.
    """
    new_files = {p: s for p, s in after.items() if p not in before}
    if not new_files:
        return []

    records: list[dict] = []
    hf_repos: dict[str, int] = {}
    other_files: list[tuple[str, int]] = []

    for path, size in new_files.items():
        repo_id = None
        for part in path.split(os.sep):
            if part.startswith("models--"):
                repo_id = part[len("models--") :].replace("--", "/", 1)
                break
        if repo_id:
            hf_repos[repo_id] = hf_repos.get(repo_id, 0) + size
        else:
            other_files.append((path, size))

    for repo_id, total_size in hf_repos.items():
        logger.debug(
            "wildedge: cache diff: HF repo=%s new_bytes=%d", repo_id, total_size
        )
        bps = int(total_size * 8 / (elapsed_ms / 1000)) if elapsed_ms > 0 else None
        records.append(
            {
                "repo_id": repo_id,
                "size": total_size,
                "duration_ms": elapsed_ms,
                "cache_hit": False,
                "bandwidth_bps": bps,
                "source_type": "huggingface",
                "source_url": f"hf://{repo_id}",
            }
        )

    for path, size in other_files:
        logger.debug("wildedge: cache diff: file=%s new_bytes=%d", path, size)
        bps = int(size * 8 / (elapsed_ms / 1000)) if elapsed_ms > 0 else None
        records.append(
            {
                "repo_id": path,
                "size": size,
                "duration_ms": elapsed_ms,
                "cache_hit": False,
                "bandwidth_bps": bps,
                "source_type": "url",
                "source_url": path,
            }
        )

    return records
