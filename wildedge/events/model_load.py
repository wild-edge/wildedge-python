from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AdapterLoad:
    adapter_id: str
    adapter_type: str
    adapter_source: str | None = None
    size_bytes: int | None = None
    rank: int | None = None
    alpha: int | None = None
    target_modules: list[str] | None = None
    load_duration_ms: int | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "adapter_id": self.adapter_id,
            "adapter_type": self.adapter_type,
        }
        for k, v in [
            ("adapter_source", self.adapter_source),
            ("size_bytes", self.size_bytes),
            ("rank", self.rank),
            ("alpha", self.alpha),
            ("target_modules", self.target_modules),
            ("load_duration_ms", self.load_duration_ms),
        ]:
            if v is not None:
                d[k] = v
        return d


@dataclass
class ModelLoadEvent:
    model_id: str
    duration_ms: int
    memory_bytes: int | None = None
    accelerator: str = "cpu"
    success: bool = True
    error_code: str | None = None
    peak_memory_bytes: int | None = None
    memory_mapped: bool | None = None
    gpu_layers: int | None = None
    threads: int | None = None
    context_length: int | None = None
    kv_cache_bytes: int | None = None
    kv_cache_quantization: str | None = None
    flash_attention: bool | None = None
    rope_scaling: str | None = None
    cold_start: bool | None = None
    compile_time_ms: int | None = None
    adapter: AdapterLoad | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    run_id: str | None = None
    agent_id: str | None = None
    step_index: int | None = None
    conversation_id: str | None = None
    context: dict[str, Any] | None = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        load_data: dict[str, Any] = {
            "duration_ms": self.duration_ms,
            "accelerator": self.accelerator,
            "success": self.success,
            "error_code": self.error_code,
        }
        for k, v in [
            ("memory_bytes", self.memory_bytes),
            ("peak_memory_bytes", self.peak_memory_bytes),
            ("memory_mapped", self.memory_mapped),
            ("gpu_layers", self.gpu_layers),
            ("threads", self.threads),
            ("context_length", self.context_length),
            ("kv_cache_bytes", self.kv_cache_bytes),
            ("kv_cache_quantization", self.kv_cache_quantization),
            ("flash_attention", self.flash_attention),
            ("rope_scaling", self.rope_scaling),
            ("cold_start", self.cold_start),
            ("compile_time_ms", self.compile_time_ms),
        ]:
            if v is not None:
                load_data[k] = v
        if self.adapter is not None:
            load_data["adapter"] = self.adapter.to_dict()

        event = {
            "event_id": self.event_id,
            "event_type": "model_load",
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "load": load_data,
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
                "attributes": self.context,
            },
        )
        return event
