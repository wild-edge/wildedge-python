from __future__ import annotations

import os
import time
import uuid
import weakref
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from wildedge import constants
from wildedge.consumer import Consumer
from wildedge.dead_letters import DeadLetterStore
from wildedge.device import DeviceInfo, detect_device
from wildedge.hubs.base import BaseHubTracker
from wildedge.hubs.huggingface import HuggingFaceHubTracker
from wildedge.hubs.registry import supported_hubs
from wildedge.hubs.torchhub import TorchHubTracker
from wildedge.integrations.base import BaseExtractor
from wildedge.integrations.gguf import GgufExtractor
from wildedge.integrations.keras import KerasExtractor
from wildedge.integrations.onnx import OnnxExtractor
from wildedge.integrations.pytorch import PytorchExtractor
from wildedge.integrations.registry import noop_integrations, supported_integrations
from wildedge.integrations.tensorflow import TensorflowExtractor
from wildedge.logging import enable_debug, logger
from wildedge.model import ModelHandle, ModelInfo, ModelRegistry
from wildedge.paths import (
    default_dead_letter_dir,
    default_model_registry_path,
    default_pending_queue_dir,
)
from wildedge.queue import EventQueue, QueuePolicy
from wildedge.reservoir import ReservoirRegistry
from wildedge.settings import read_client_env, resolve_app_identity
from wildedge.timing import Timer, elapsed_ms
from wildedge.transmitter import Transmitter

DSN_FORMAT = "'https://<project-secret>@ingest.wildedge.dev/<project-key>'"
ERROR_DSN_MISSING_SECRET = f"DSN must include a project secret: {DSN_FORMAT}"
ERROR_DSN_REQUIRED = (
    f"DSN is required. Pass dsn= or set {constants.ENV_DSN}. Format: {DSN_FORMAT}"
)
ERROR_BATCH_SIZE_RANGE = f"batch_size must be between {constants.BATCH_SIZE_MIN} and {constants.BATCH_SIZE_MAX}"
ERROR_FLUSH_INTERVAL_RANGE = (
    "flush_interval_sec must be between "
    f"{constants.FLUSH_INTERVAL_MIN} and {constants.FLUSH_INTERVAL_MAX}"
)
ERROR_MAX_QUEUE_SIZE_RANGE = (
    "max_queue_size must be between "
    f"{constants.MAX_QUEUE_SIZE_MIN} and {constants.MAX_QUEUE_SIZE_MAX}"
)
ERROR_MAX_EVENT_AGE = "max_event_age_sec must be greater than 0"
ERROR_MAX_DEAD_LETTER_BATCHES = "max_dead_letter_batches must be >= 0"
ERROR_UNKNOWN_INTEGRATION = (
    "Unknown integration {integration!r}. Available integrations: {integrations}; "
    "available hubs: {hubs}"
)
LOG_REGISTERED_MODEL = "wildedge: registered model id=%s format=%s"
LOG_INSTRUMENT_TORCH_KERAS = (
    "wildedge: instrument(%r) - inference hooks fire automatically "
    "on register_model(); use client.load() for load/unload tracking"
)


def parse_dsn(dsn: str) -> tuple[str, str, str]:
    """Parse DSN into (secret, host, project_key)."""
    parsed = urlparse(dsn)
    if not parsed.username:
        raise ValueError(ERROR_DSN_MISSING_SECRET)
    project_key = parsed.path.lstrip("/").split("/", 1)[0]
    if not project_key:
        raise ValueError(f"DSN must include a project key path: {DSN_FORMAT}")
    host = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        host += f":{parsed.port}"
    return parsed.username, host, project_key


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

    # Framework integrations: patch ML runtime constructors for inference tracking.
    SUPPORTED_INTEGRATIONS = supported_integrations()
    NOOP_INTEGRATIONS = noop_integrations()
    PATCH_INSTALLERS = {
        "gguf": GgufExtractor.install_auto_load_patch,
        "onnx": OnnxExtractor.install_auto_load_patch,
        "timm": PytorchExtractor.install_timm_patch,
        "tensorflow": TensorflowExtractor.install_auto_load_patch,
    }

    # Hub trackers: record download provenance (where models came from).
    SUPPORTED_HUBS = supported_hubs()
    HUB_TRACKER_CLASSES: dict[str, type[BaseHubTracker]] = {
        "huggingface": HuggingFaceHubTracker,
        "torchhub": TorchHubTracker,
    }

    def __init__(
        self,
        *,
        dsn: str | None = None,
        app_version: str | None = None,
        device: DeviceInfo | None = None,
        queue_policy: QueuePolicy = QueuePolicy.OPPORTUNISTIC,
        max_queue_size: int = constants.DEFAULT_MAX_QUEUE_SIZE,
        batch_size: int = constants.DEFAULT_BATCH_SIZE,
        flush_interval_sec: float = constants.DEFAULT_FLUSH_INTERVAL_SEC,
        debug: bool | None = None,
        max_event_age_sec: float = constants.DEFAULT_MAX_EVENT_AGE_SEC,
        enable_offline_persistence: bool = constants.DEFAULT_ENABLE_OFFLINE_PERSISTENCE,
        app_identity: str | None = None,
        offline_queue_dir: str | None = None,
        enable_dead_letter_persistence: bool = (
            constants.DEFAULT_ENABLE_DEAD_LETTER_PERSISTENCE
        ),
        dead_letter_dir: str | None = None,
        max_dead_letter_batches: int = constants.DEFAULT_MAX_DEAD_LETTER_BATCHES,
        on_delivery_failure: Callable[[str, int, int], None] | None = None,
        reservoir_size: int = constants.DEFAULT_RESERVOIR_SIZE,
        low_confidence_threshold: float = constants.DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        high_entropy_threshold: float = constants.DEFAULT_HIGH_ENTROPY_THRESHOLD,
        low_confidence_slots_pct: float = constants.DEFAULT_LOW_CONFIDENCE_SLOTS_PCT,
        priority_fn: Callable[[dict], bool] | None = None,
    ):
        env = read_client_env(dsn=dsn, debug=debug, app_identity=app_identity)
        dsn = env.dsn
        if not dsn:
            raise ValueError(ERROR_DSN_REQUIRED)
        api_key, host, project_key = parse_dsn(dsn)
        debug = env.debug
        app_identity = resolve_app_identity(
            explicit=env.app_identity,
            project_key=project_key,
        )

        # Validate configuration ranges
        if not (constants.BATCH_SIZE_MIN <= batch_size <= constants.BATCH_SIZE_MAX):
            raise ValueError(ERROR_BATCH_SIZE_RANGE)
        if not (
            constants.FLUSH_INTERVAL_MIN
            <= flush_interval_sec
            <= constants.FLUSH_INTERVAL_MAX
        ):
            raise ValueError(ERROR_FLUSH_INTERVAL_RANGE)
        if not (
            constants.MAX_QUEUE_SIZE_MIN
            <= max_queue_size
            <= constants.MAX_QUEUE_SIZE_MAX
        ):
            raise ValueError(ERROR_MAX_QUEUE_SIZE_RANGE)
        if max_event_age_sec <= 0:
            raise ValueError(ERROR_MAX_EVENT_AGE)
        if max_dead_letter_batches < 0:
            raise ValueError(ERROR_MAX_DEAD_LETTER_BATCHES)

        self.api_key = api_key
        self.debug = debug
        self.closed = False

        if debug:
            enable_debug()
            logger.debug("wildedge: debug mode enabled")

        self.device = device or detect_device(api_key=api_key, app_version=app_version)
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc)
        resolved_offline_queue_dir = offline_queue_dir or str(
            default_pending_queue_dir(app_identity)
        )
        resolved_dead_letter_dir = dead_letter_dir or str(
            default_dead_letter_dir(app_identity)
        )
        resolved_model_registry_path = str(default_model_registry_path(app_identity))
        self.queue = EventQueue(
            max_size=max_queue_size,
            policy=queue_policy,
            persist_to_disk=enable_offline_persistence,
            disk_dir=resolved_offline_queue_dir,
        )
        if debug and self.queue.length() > 0:
            logger.debug(
                "wildedge: loaded %d offline event(s) from previous session",
                self.queue.length(),
            )
        self.registry = ModelRegistry(
            persist_path=resolved_model_registry_path
            if enable_offline_persistence
            else None
        )
        self.transmitter = Transmitter(api_key=api_key, host=host)
        self.dead_letter_store = DeadLetterStore(
            enabled=enable_dead_letter_persistence,
            directory=resolved_dead_letter_dir,
            max_batches=max_dead_letter_batches,
        )
        self.reservoir_registry = ReservoirRegistry(
            reservoir_size=reservoir_size,
            low_confidence_threshold=low_confidence_threshold,
            high_entropy_threshold=high_entropy_threshold,
            low_confidence_slots_pct=low_confidence_slots_pct,
            priority_fn=priority_fn,
        )
        self.consumer = Consumer(
            queue=self.queue,
            transmitter=self.transmitter,
            device=self.device,
            get_models=self.registry.snapshot,
            session_id=self.session_id,
            batch_size=batch_size,
            flush_interval_sec=flush_interval_sec,
            debug=debug,
            max_event_age_sec=max_event_age_sec,
            dead_letter_store=self.dead_letter_store,
            on_delivery_failure=on_delivery_failure,
            reservoir_registry=self.reservoir_registry,
        )

        self.auto_loaded: set[str] = set()
        # Active hub trackers keyed by hub name. Populated by _activate_hub()
        # when instrument() is called with a hub name.
        self.hub_trackers: dict[str, BaseHubTracker] = {}

        if hasattr(os, "register_at_fork"):
            os.register_at_fork(
                before=self.consumer._pause,
                after_in_child=self.consumer._resume,
                after_in_parent=self.consumer._resume,
            )

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

        if event_dict.get("event_type") == "inference":
            model_id = event_dict.get("model_id", "")
            self.reservoir_registry.get_or_create(model_id).add(event_dict)
        else:
            event_dict.setdefault("__we_first_queued_at", time.time())
            event_dict.setdefault("__we_attempts", 0)
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

    def _activate_hub(self, hub_name: str) -> None:
        """Instantiate and install the patch for a hub tracker (idempotent)."""
        if hub_name in self.hub_trackers:
            return
        tracker = self.HUB_TRACKER_CLASSES[hub_name]()
        if not tracker.can_install():
            logger.warning(
                "wildedge: hub '%s' library not installed; skipping", hub_name
            )
            return
        tracker.install_patch(weakref.ref(self))
        self.hub_trackers[hub_name] = tracker
        if self.debug:
            logger.debug("wildedge: hub tracker '%s' activated", hub_name)

    def _snapshot_hub_caches(self) -> dict[str, dict[str, int]]:
        """Return a before-snapshot of all active hub cache directories."""
        return {
            name: tracker.scan_cache() for name, tracker in self.hub_trackers.items()
        }

    def _diff_hub_caches(
        self, before: dict[str, dict[str, int]], load_ms: int
    ) -> list[dict]:
        """
        Compute download records from a before/after hub cache diff.

        Calls each active hub tracker's ``diff_to_records`` against its
        corresponding before snapshot, merging all results.
        """
        records: list[dict] = []
        for name, tracker in self.hub_trackers.items():
            after = tracker.scan_cache()
            records.extend(
                tracker.diff_to_records(before.get(name, {}), after, load_ms)
            )
        return records

    def _drain_hub_trackers(self) -> list[dict]:
        """Drain and return thread-local download records from all active hub trackers."""
        records: list[dict] = []
        for tracker in self.hub_trackers.values():
            records.extend(tracker.drain())
        return records

    def instrument(
        self, integration: str | None, *, hubs: list[str] | None = None
    ) -> None:
        """
        Activate auto-tracking for a framework integration, hub trackers, or both.

        Framework integrations patch ML runtime constructors so that models
        created afterwards are registered and timed automatically.  Hub trackers
        record download provenance (where the model came from, cache hits,
        bandwidth).

        Framework + hub tracking
        ------------------------
        Pass ``hubs=`` alongside a framework name to activate download provenance
        tracking for the hub(s) your code downloads from::

            client.instrument("gguf", hubs=["huggingface"])
            client.instrument("onnx", hubs=["huggingface"])
            client.instrument("timm", hubs=["huggingface", "torchhub"])

        Hub tracking only (no framework)
        ---------------------------------
        Pass ``None`` as the integration to activate hub trackers without
        instrumenting any framework::

            client.instrument(None, hubs=["huggingface"])
            client.instrument(None, hubs=["torchhub"])

        Supported framework integrations
        ---------------------------------
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
            Inference is tracked automatically once a model is registered;
            use ``client.load(MyModel)`` for load/unload timing.

        Supported hubs
        --------------
        ``"huggingface"``
            Patches ``hf_hub_download`` and ``snapshot_download`` for
            thread-local download tracking.  Also enables filesystem-diff
            support for timm models that pull from HuggingFace Hub.
            Requires ``huggingface-hub``.
        ``"torchhub"``
            Patches ``torch.hub.load`` and scans the torch hub cache for
            files downloaded via ``torch.hub.download_url_to_file``.
            Requires ``torch``.

        Each integration or hub tracker is installed at most once per process
        regardless of how many times ``instrument()`` is called.
        """
        if integration is None:
            if not hubs:
                raise ValueError(
                    "instrument(None) requires hubs= to be non-empty; "
                    "pass the hub name(s) you want to activate, "
                    "e.g. instrument(None, hubs=['huggingface'])"
                )
            unknown = [h for h in hubs if h not in self.SUPPORTED_HUBS]
            if unknown:
                raise ValueError(
                    f"Unknown hub(s): {unknown}. "
                    f"Available hubs: {sorted(self.SUPPORTED_HUBS)}"
                )
            for hub_name in hubs:
                self._activate_hub(hub_name)
            return

        if integration in self.SUPPORTED_HUBS:
            raise ValueError(
                f"{integration!r} is a hub, not a framework integration. "
                f"Use instrument(None, hubs=[{integration!r}]) to activate it."
            )
        if integration not in self.SUPPORTED_INTEGRATIONS:
            raise ValueError(
                ERROR_UNKNOWN_INTEGRATION.format(
                    integration=integration,
                    integrations=sorted(self.SUPPORTED_INTEGRATIONS),
                    hubs=sorted(self.SUPPORTED_HUBS),
                )
            )
        if hubs:
            unknown = [h for h in hubs if h not in self.SUPPORTED_HUBS]
            if unknown:
                raise ValueError(
                    f"Unknown hub(s): {unknown}. "
                    f"Available hubs: {sorted(self.SUPPORTED_HUBS)}"
                )
        if integration in self.NOOP_INTEGRATIONS:
            # Models are user-defined subclasses; no global constructor to patch.
            # Inference is tracked automatically once a model is registered via
            # client.load() or register_model(); load/unload requires client.load().
            if self.debug:
                logger.debug(LOG_INSTRUMENT_TORCH_KERAS, integration)
        else:
            installer = self.PATCH_INSTALLERS[integration]
            installer(weakref.ref(self))
        for hub_name in hubs or []:
            self._activate_hub(hub_name)

    def _on_model_auto_loaded(
        self,
        obj: object,
        *,
        load_ms: int,
        downloads: list[dict] | None = None,
        model_id: str | None = None,
        load_kwargs: dict | None = None,
    ) -> None:
        """Callback invoked by auto-patches after a model is constructed."""
        if self.closed:
            return
        # Always drain thread-local hub buffers to keep them clean.
        # Prefer caller-supplied downloads (timm/torchhub filesystem diff) over
        # thread-local records. If both exist, the diff wins and thread records
        # are discarded to avoid double-reporting.
        thread_records = self._drain_hub_trackers()
        if downloads is None and thread_records:
            downloads = thread_records

        handle = self.register_model(obj, model_id=model_id)
        self.auto_loaded.add(handle.model_id)

        memory = self._memory_bytes_for(obj)

        # Emit one download event per repo_id, aggregating per-file records.
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

        # If an auto-patch already registered and tracked this model's load
        # (e.g. the user called client.load(Llama, ...) despite the patch being
        # active), skip the duplicate tracking.
        if handle.model_id in self.auto_loaded:
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

    def close(self, timeout: float | None = None) -> None:
        """Best-effort shutdown; pass timeout to attempt bounded flush first."""
        self.closed = True
        if timeout is None:
            self.consumer.close()
        else:
            self.consumer.close(timeout=timeout)

    def pending_count(self) -> int:
        """Return the number of events currently buffered."""
        return self.queue.length()

    def __enter__(self) -> WildEdge:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
