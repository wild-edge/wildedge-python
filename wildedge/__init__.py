"""WildEdge Python SDK."""

from wildedge.attachments import Attachment
from wildedge.client import SpanContextManager, WildEdge
from wildedge.convenience import init
from wildedge.decorators import track
from wildedge.defaults import (
    flush,
    get_client,
    register_model,
    set_default_client,
    span,
    trace,
    track_span,
)
from wildedge.events import (
    AdapterDownload,
    AdapterLoad,
    AudioInputMeta,
    ClassificationOutputMeta,
    DetectionOutputMeta,
    EmbeddingOutputMeta,
    ErrorCode,
    FeedbackType,
    GenerationConfig,
    GenerationOutputMeta,
    ImageInputMeta,
    SpanEvent,
    TextInputMeta,
)
from wildedge.events.span import SpanKind, SpanStatus
from wildedge.llm_api import LLMCall, llm_api
from wildedge.platforms import capture_hardware
from wildedge.platforms.device_info import DeviceInfo
from wildedge.platforms.hardware import HardwareContext, ThermalContext
from wildedge.queue import QueuePolicy
from wildedge.timing import Timer
from wildedge.tracing import (
    SpanContext,
    TraceContext,
    get_span_context,
    get_trace_context,
    span_context,
)

__all__ = [
    "WildEdge",
    "init",
    "get_client",
    "set_default_client",
    "trace",
    "span",
    "track_span",
    "register_model",
    "flush",
    "llm_api",
    "LLMCall",
    "Attachment",
    "capture_hardware",
    "HardwareContext",
    "ThermalContext",
    "track",
    "QueuePolicy",
    "DeviceInfo",
    "ImageInputMeta",
    "AudioInputMeta",
    "TextInputMeta",
    "DetectionOutputMeta",
    "ClassificationOutputMeta",
    "GenerationOutputMeta",
    "EmbeddingOutputMeta",
    "GenerationConfig",
    "AdapterLoad",
    "AdapterDownload",
    "SpanEvent",
    "FeedbackType",
    "ErrorCode",
    "Timer",
    "span_context",
    "TraceContext",
    "SpanContext",
    "get_trace_context",
    "get_span_context",
    "SpanKind",
    "SpanStatus",
    "SpanContextManager",
]
