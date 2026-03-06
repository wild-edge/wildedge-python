from __future__ import annotations

import re
from pathlib import Path

from wildedge.platforms import CURRENT_PLATFORM


def normalize_namespace(namespace: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", namespace.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "default"


def default_sdk_state_dir() -> Path:
    return CURRENT_PLATFORM.state_base() / "wildedge"


def default_sdk_cache_dir() -> Path:
    return CURRENT_PLATFORM.cache_base() / "wildedge"


def default_pending_queue_dir(namespace: str = "default") -> Path:
    return default_sdk_state_dir() / normalize_namespace(namespace) / "pending_queue"


def default_model_registry_path(namespace: str = "default") -> Path:
    return (
        default_sdk_state_dir() / normalize_namespace(namespace) / "model_registry.json"
    )


def default_dead_letter_dir(namespace: str = "default") -> Path:
    return default_sdk_cache_dir() / normalize_namespace(namespace) / "dead_letters"
