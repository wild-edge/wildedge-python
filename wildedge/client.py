from __future__ import annotations

import os
import time
import uuid
import weakref
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from wildedge import config
from wildedge.consumer import Consumer
from wildedge.device import DeviceInfo, detect_device
from wildedge.integrations.base import BaseExtractor
from wildedge.integrations.gguf import GgufExtractor
from wildedge.integrations.hf import drain_downloads
from wildedge.integrations.hf import install_patch as _hf_install_patch
from wildedge.integrations.keras import KerasExtractor
from wildedge.integrations.onnx import OnnxExtractor
from wildedge.integrations.pytorch import PytorchExtractor
from wildedge.integrations.tensorflow import TensorflowExtractor
from wildedge.logging import enable_debug, logger
from wildedge.model import ModelHandle, ModelInfo, ModelRegistry
from wildedge.queue import EventQueue, QueuePolicy
from wildedge.timing import Timer, elapsed_ms
from wildedge.transmitter import Transmitter

DSN_FORMAT = "'https://<project-secret>@ingest.wildedge.dev/<project-key>'"
ERROR_DSN_MISSING_SECRET = f"DSN must include a project secret: {DSN_FORMAT}"
ERROR_DSN_REQUIRED = (
    f"DSN is required. Pass dsn= or set {config.ENV_DSN}. Format: {DSN_FORMAT}"
)
ERROR_BATCH_SIZE_RANGE = (
    f"batch_size must be between {config.BATCH_SIZE_MIN} and {config.BATCH_SIZE_MAX}"
)
ERROR_FLUSH_INTERVAL_RANGE = (
    "flush_interval_sec must be between "
    f"{config.FLUSH_INTERVAL_MIN} and {config.FLUSH_INTERVAL_MAX}"
)
ERROR_MAX_QUEUE_SIZE_RANGE = (
    "max_queue_size must be between "
    f"{config.MAX_QUEUE_SIZE_MIN} and {config.MAX_QUEUE_SIZE_MAX}"
)
ERROR_UNKNOWN_INTEGRATION = (
    "Unknown integration {integration!r}. Available: {available}"
)
LOG_REGISTERED_MODEL = "wildedge: registered model id=%s format=%s"
LOG_INSTRUMENT_TORCH_KERAS = (
    "wildedge: instrument(%r) - inference hooks fire automatically "
    "on register_model(); use client.load() for load/unload tracking"
)
NOOP_INTEGRATIONS = {"torch", "keras"}


def parse_dsn(dsn: str) -> tuple[str, str]:
    """Parse 'https://<project-secret>@ingest.wildedge.dev/<project-key>' → (secret, host)."""
    parsed = urlparse(dsn)
    if not parsed.username:
        raise ValueError(ERROR_DSN_MISSING_SECRET)
    host = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        host += f":{parsed.port}"
    return parsed.username, host


DEFAULT_EXTRACTORS: list[BaseExtractor] = [
    OnnxExtractor(),
    GgufExtractor(),
    PytorchExtractor(),
    TensorflowExtractor(),
    KerasExtractor(),
]


class WildEdge:
    """
    WildEdge on-device ML monitoring client.

    Usage::

        client = WildEdge(dsn="https://<secret>@ingest.wildedge.dev/<key>", app_version="1.0.0")
        session = ort.InferenceSession("model.onnx")
        handle = client.register_model(session)
        # inference is now tracked automatically
    """

    SUPPORTED_INTEGRATIONS = {
        "gguf",
        "onnx",
        "timm",
        "torch",
        "keras",
        "tensorflow",
        "huggingface",
    }
    PATCH_INSTALLERS = {
        "gguf": GgufExtractor.install_auto_load_patch,
        "onnx": OnnxExtractor.install_auto_load_patch,
        "timm": PytorchExtractor.install_timm_patch,
        "tensorflow": TensorflowExtractor.install_auto_load_patch,
    }

    def __init__(
        self,
        *,
        dsn: str | None = None,
        app_version: str | None = None,
        device: DeviceInfo | None = None,
        queue_policy: QueuePolicy = QueuePolicy.OPPORTUNISTIC,
        max_queue_size: int = config.DEFAULT_MAX_QUEUE_SIZE,
        batch_size: int = config.DEFAULT_BATCH_SIZE,
        flush_interval_sec: float = config.DEFAULT_FLUSH_INTERVAL_SEC,
        debug: bool | None = None,
    ):
        dsn = dsn or os.environ.get(config.ENV_DSN)
        if not dsn:
            raise ValueError(ERROR_DSN_REQUIRED)
        api_key, host = parse_dsn(dsn)
        if debug is None:
            debug = os.environ.get(config.ENV_DEBUG, "").lower() in ("1", "true", "yes")

        # Validate configuration ranges
        if not (config.BATCH_SIZE_MIN <= batch_size <= config.BATCH_SIZE_MAX):
            raise ValueError(ERROR_BATCH_SIZE_RANGE)
        if not (
            config.FLUSH_INTERVAL_MIN <= flush_interval_sec <= config.FLUSH_INTERVAL_MAX
        ):
            raise ValueError(ERROR_FLUSH_INTERVAL_RANGE)
        if not (
            config.MAX_QUEUE_SIZE_MIN <= max_queue_size <= config.MAX_QUEUE_SIZE_MAX
        ):
            raise ValueError(ERROR_MAX_QUEUE_SIZE_RANGE)

        self.api_key = api_key
        self.debug = debug
        self.closed = False

        if debug:
            enable_debug()
            logger.debug("wildedge: debug mode enabled")

        self.device = device or detect_device(api_key=api_key, app_version=app_version)
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc)
        self.queue = EventQueue(max_size=max_queue_size, policy=queue_policy)
        self.registry = ModelRegistry()
        self.transmitter = Transmitter(api_key=api_key, host=host)
        self.consumer = Consumer(
            queue=self.queue,
            transmitter=self.transmitter,
            device=self.device,
            get_models=self.registry.snapshot,
            session_id=self.session_id,
            batch_size=batch_size,
            flush_interval_sec=flush_interval_sec,
            debug=debug,
        )

        self._auto_loaded: set[str] = set()
        self._hf_instrumented: bool = False

        if debug:
            logger.debug("wildedge: client initialized (session=%s)", self.session_id)

    def publish(self, event_dict: dict) -> None:
        if self.closed:
            return

        if self.debug:
            logger.debug(
                "wildedge: queuing event type=%s model=%s",
                event_dict.get("event_type"),
                event_dict.get("model_id"),
            )
        self.queue.add(event_dict)

    def register_model(
        self,
        model_obj: object,
        *,
        model_id: str | None = None,
        source: str | None = None,
        family: str | None = None,
        version: str | None = None,
        quantization: str | None = None,
        auto_instrument: bool = True,
    ) -> ModelHandle:
        """
        Register a model and return a handle for tracking events.

        Auto-extracts metadata from ONNX Runtime and GGUF/llama.cpp objects.
        User-supplied kwargs override extracted values.
        """
        overrides = {
            k: v
            for k, v in {
                "id": model_id,
                "source": source,
                "family": family,
                "version": version,
                "quantization": quantization,
            }.items()
            if v is not None
        }

        extractor = self._find_extractor(model_obj)

        if extractor is not None:
            model_id, info = extractor.extract_info(model_obj, overrides)
        else:
            # No extractor matched - require explicit id
            model_id = overrides.pop("id", None)
            model_name = overrides.pop("model_name", None) or (
                str(type(model_obj).__name__)
            )
            info = ModelInfo(
                model_name=model_name,
                model_version=overrides.pop("version", "unknown"),
                model_source=overrides.pop("source", "local"),
                model_format=overrides.pop("format", "unknown"),
                model_family=overrides.pop("family", None),
                quantization=overrides.pop("quantization", None),
            )

        if model_id is None:
            raise ValueError(
                "Could not auto-derive a stable model_id for this model type. "
                "Pass id='your-model-id' to register_model()."
            )

        handle, is_new = self.registry.register(model_id, info, self.publish)

        if not is_new:
            logger.debug(
                "wildedge: model '%s' already registered - returning existing handle",
                model_id,
            )
            return handle

        if auto_instrument and extractor is not None:
            extractor.install_hooks(model_obj, handle)

        if self.debug:
            logger.debug(
                LOG_REGISTERED_MODEL,
                model_id,
                info.model_format,
            )

        return handle

    def _find_extractor(self, model_obj: object) -> BaseExtractor | None:
        for candidate in DEFAULT_EXTRACTORS:
            if candidate.can_handle(model_obj):
                return candidate
        return None

    def _memory_bytes_for(self, model_obj: object) -> int | None:
        extractor = self._find_extractor(model_obj)
        if extractor is None:
            return None
        return extractor.memory_bytes(model_obj)

    def _instrument_huggingface(self) -> None:
        _hf_install_patch()
        self._hf_instrumented = True

    def instrument(self, integration: str) -> None:
        """
        Activate load/unload auto-tracking for a supported library.

        Patches the library's constructor or factory function so that models
        created afterwards are registered and timed automatically with no
        ``client.load()`` or ``client.register_model()`` call needed.

        Supported values
        ----------------
        ``"gguf"``
            Patches ``llama_cpp.Llama.__init__``. Requires ``llama-cpp-python``.
        ``"onnx"``
            Replaces ``ort.InferenceSession`` with a timed subclass.
            Requires ``onnxruntime``.
        ``"timm"``
            Patches ``timm.create_model``. Requires ``timm``.
        ``"tensorflow"``
            Patches ``tf.keras.models.load_model`` and ``tf.saved_model.load``.
            Requires ``tensorflow``.
        ``"torch"`` / ``"keras"``
            No global constructor to patch; models are user-defined subclasses.
            This call succeeds and is a no-op: inference is tracked automatically
            once a model is registered; use ``client.load(MyModel)`` for
            load/unload tracking.

        Each integration is installed at most once per process regardless of
        how many times ``instrument()`` is called.

        Usage::

            client = WildEdge(dsn="https://<project-secret>@ingest.wildedge.dev/<project-key>", app_version="1.0.0")
            client.instrument("gguf")
            client.instrument("onnx")
            client.instrument("timm")
        """
        if integration not in self.SUPPORTED_INTEGRATIONS:
            raise ValueError(
                ERROR_UNKNOWN_INTEGRATION.format(
                    integration=integration,
                    available=sorted(self.SUPPORTED_INTEGRATIONS),
                )
            )
        if integration in NOOP_INTEGRATIONS:
            # Models are user-defined subclasses; no global constructor to patch.
            # Inference is tracked automatically once a model is registered via
            # client.load() or register_model(); load/unload requires client.load().
            if self.debug:
                logger.debug(
                    LOG_INSTRUMENT_TORCH_KERAS,
                    integration,
                )
            return
        if integration == "huggingface":
            self._instrument_huggingface()
            return
        installer = self.PATCH_INSTALLERS[integration]
        installer(weakref.ref(self))

    def _on_model_auto_loaded(
        self,
        obj: object,
        *,
        load_ms: int,
        downloads: list[dict] | None = None,
        model_id: str | None = None,
        load_kwargs: dict | None = None,
    ) -> None:
        """Callback from OTel-style auto-patches after a model is constructed."""
        if self.closed:
            return
        # Drain HF buffer: prefer caller-supplied downloads (timm cache diff);
        # fall back to thread-local HF records (ONNX + explicit hf_hub_download).
        # Always drain to keep the buffer clean even when not used.
        hf_records = drain_downloads() if self._hf_instrumented else []
        if downloads is None and hf_records:
            downloads = hf_records
        handle = self.register_model(obj, model_id=model_id)
        self._auto_loaded.add(handle.model_id)

        memory = self._memory_bytes_for(obj)

        # Emit download event before load. Aggregate all per-file HF Hub records
        # into a single event grouped by repo_id.
        if downloads:
            repos: dict[str, list[dict]] = {}
            for rec in downloads:
                repos.setdefault(rec["repo_id"], []).append(rec)
            for repo_id, recs in repos.items():
                total_size = sum(r["size"] for r in recs)
                total_duration = sum(r["duration_ms"] for r in recs)
                all_cached = all(r["cache_hit"] for r in recs)
                source_type = recs[0].get("source_type", "huggingface")
                source_url = recs[0].get("source_url", f"hf://{repo_id}")
                bps_vals = [r["bandwidth_bps"] for r in recs if r.get("bandwidth_bps")]
                bandwidth_bps = int(sum(bps_vals) / len(bps_vals)) if bps_vals else None
                handle.track_download(
                    source_url=source_url,
                    source_type=source_type,
                    file_size_bytes=total_size,
                    downloaded_bytes=0 if all_cached else total_size,
                    duration_ms=total_duration,
                    bandwidth_bps=bandwidth_bps,
                    network_type="unknown",
                    resumed=False,
                    cache_hit=all_cached,
                    success=True,
                )

        handle.track_load(
            duration_ms=load_ms, memory_bytes=memory, **(load_kwargs or {})
        )

        loaded_at = time.perf_counter()

        def _on_unload() -> None:
            handle.track_unload(
                duration_ms=0, reason="gc", uptime_ms=elapsed_ms(loaded_at)
            )

        weakref.finalize(obj, _on_unload)

    def load(self, model_class: type, *args: Any, **kwargs: Any) -> object:
        """
        Instantiate a model, register it, and track load/unload automatically.

        For GGUF (llama.cpp) and timm models the WildEdge client patches the
        constructor/factory at initialisation time, so plain ``Llama(...)`` or
        ``timm.create_model(...)`` calls are tracked without using this method.
        ``load()`` is intended for other model types (e.g. custom PyTorch modules).

        Usage::

            model = client.load(MyTorchModel)
            session = client.load(ort.InferenceSession, "model.onnx")
        """
        with Timer() as t:
            obj = model_class(*args, **kwargs)
        load_ms = t.elapsed_ms

        handle = self.register_model(obj)

        # If an OTel auto-patch already registered and tracked this model's load
        # (e.g. the user called client.load(Llama, ...) despite the patch being
        # active), skip the duplicate tracking.
        if handle.model_id in self._auto_loaded:
            return obj

        memory = self._memory_bytes_for(obj)

        handle.track_load(duration_ms=load_ms, memory_bytes=memory)

        loaded_at = time.perf_counter()

        def _on_unload() -> None:
            handle.track_unload(
                duration_ms=0, reason="gc", uptime_ms=elapsed_ms(loaded_at)
            )

        weakref.finalize(obj, _on_unload)

        return obj

    def flush(self, timeout: float = 5.0) -> None:
        """Block until the event queue drains or timeout expires."""
        self.consumer.flush(timeout=timeout)

    def close(self) -> None:
        """Flush remaining events and stop the consumer thread."""
        self.closed = True
        self.consumer.close()

    def pending_count(self) -> int:
        """Return the number of events currently buffered."""
        return self.queue.length()

    def __enter__(self) -> WildEdge:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
