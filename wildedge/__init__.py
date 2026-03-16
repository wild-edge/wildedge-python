"""WildEdge Python SDK."""

from wildedge.client import WildEdge
from wildedge.decorators import track
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
    TextInputMeta,
)
from wildedge.platforms import capture_hardware
from wildedge.platforms.device_info import DeviceInfo
from wildedge.platforms.hardware import HardwareContext, ThermalContext
from wildedge.queue import QueuePolicy
from wildedge.timing import Timer

__all__ = [
    "WildEdge",
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
    "FeedbackType",
    "ErrorCode",
    "Timer",
]
