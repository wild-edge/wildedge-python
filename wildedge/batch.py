from __future__ import annotations

import uuid
from datetime import datetime, timezone

from wildedge import constants
from wildedge.device import DeviceInfo


def _sanitize_event(event: dict) -> dict:
    return {k: v for k, v in event.items() if not k.startswith("__we_")}


def build_batch(
    device: DeviceInfo,
    models: dict[str, dict],
    events: list[dict],
    session_id: str,
    created_at: datetime,
    sampling: dict | None = None,
) -> dict:
    """Build a protocol-compliant batch envelope."""
    batch: dict = {
        "protocol_version": constants.PROTOCOL_VERSION,
        "device": device.to_dict(),
        "models": models,
        "session_id": session_id,
        "batch_id": str(uuid.uuid4()),
        "created_at": created_at.isoformat(),
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }
    if sampling is not None:
        batch["sampling"] = sampling
    batch["events"] = [_sanitize_event(event) for event in events]
    return batch
