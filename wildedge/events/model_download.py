from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AdapterDownload:
    adapter_id: str
    adapter_type: str
    for_base_model: str
    rank: int | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "adapter_id": self.adapter_id,
            "adapter_type": self.adapter_type,
            "for_base_model": self.for_base_model,
        }
        if self.rank is not None:
            d["rank"] = self.rank
        return d


@dataclass
class ModelDownloadEvent:
    model_id: str
    source_url: str
    source_type: str
    file_size_bytes: int
    downloaded_bytes: int
    duration_ms: int
    network_type: str
    resumed: bool
    cache_hit: bool
    success: bool
    bandwidth_bps: int | None = None
    network_generation: str | None = None
    resume_offset_bytes: int | None = None
    retry_count: int | None = None
    checksum_verified: bool | None = None
    checksum_algorithm: str | None = None
    decompression_time_ms: int | None = None
    storage_type: str | None = None
    storage_available_bytes: int | None = None
    http_status: int | None = None
    cdn_edge: str | None = None
    error_code: str | None = None
    adapter: AdapterDownload | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    run_id: str | None = None
    agent_id: str | None = None
    step_index: int | None = None
    conversation_id: str | None = None
    attributes: dict[str, Any] | None = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        download_data: dict[str, Any] = {
            "source_url": self.source_url,
            "source_type": self.source_type,
            "file_size_bytes": self.file_size_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "duration_ms": self.duration_ms,
            "network_type": self.network_type,
            "resumed": self.resumed,
            "cache_hit": self.cache_hit,
            "success": self.success,
            "error_code": self.error_code,
        }
        for k, v in [
            ("bandwidth_bps", self.bandwidth_bps),
            ("network_generation", self.network_generation),
            ("resume_offset_bytes", self.resume_offset_bytes),
            ("retry_count", self.retry_count),
            ("checksum_verified", self.checksum_verified),
            ("checksum_algorithm", self.checksum_algorithm),
            ("decompression_time_ms", self.decompression_time_ms),
            ("storage_type", self.storage_type),
            ("storage_available_bytes", self.storage_available_bytes),
            ("http_status", self.http_status),
            ("cdn_edge", self.cdn_edge),
        ]:
            if v is not None:
                download_data[k] = v
        if self.adapter is not None:
            download_data["adapter"] = self.adapter.to_dict()

        event = {
            "event_id": self.event_id,
            "event_type": "model_download",
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "download": download_data,
        }
        from wildedge.events.common import add_optional_fields

        add_optional_fields(
            event,
            {
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "parent_span_id": self.parent_span_id,
                "run_id": self.run_id,
                "agent_id": self.agent_id,
                "step_index": self.step_index,
                "conversation_id": self.conversation_id,
                "attributes": self.attributes,
            },
        )
        return event
