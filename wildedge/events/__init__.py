"""Public event classes and metadata types."""

from wildedge.events.error import ErrorCode, ErrorEvent
from wildedge.events.feedback import FeedbackEvent, FeedbackType
from wildedge.events.inference import (
    AudioInputMeta,
    ClassificationOutputMeta,
    DetectionOutputMeta,
    EmbeddingOutputMeta,
    GenerationConfig,
    GenerationOutputMeta,
    HistogramSummary,
    ImageInputMeta,
    InferenceEvent,
    TextInputMeta,
    TopKPrediction,
)
from wildedge.events.model_download import AdapterDownload, ModelDownloadEvent
from wildedge.events.model_load import AdapterLoad, ModelLoadEvent
from wildedge.events.model_unload import ModelUnloadEvent

__all__ = [
    "AdapterDownload",
    "AdapterLoad",
    "AudioInputMeta",
    "ClassificationOutputMeta",
    "DetectionOutputMeta",
    "EmbeddingOutputMeta",
    "ErrorCode",
    "ErrorEvent",
    "FeedbackEvent",
    "FeedbackType",
    "GenerationConfig",
    "GenerationOutputMeta",
    "HistogramSummary",
    "ImageInputMeta",
    "InferenceEvent",
    "ModelDownloadEvent",
    "ModelLoadEvent",
    "ModelUnloadEvent",
    "TextInputMeta",
    "TopKPrediction",
]
