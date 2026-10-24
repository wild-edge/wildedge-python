"""Framework integration registry.

Contains only ML *framework* integrations (inference tracking, load/unload
timing).  Model hub and repository trackers (download provenance) live in
``wildedge.hubs.registry``. They have different activation semantics.

Adapters are listed by name and imported only when used, so importing the SDK
does not import any ML framework (see ``wildedge.loading``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import cache
from typing import Literal

from wildedge.loading import resolve

IntegrationKind = Literal["noop", "client_patch"]

ADAPTER_PACKAGE = "wildedge.integrations"


@dataclass(frozen=True)
class IntegrationSpec:
    """Describes one integration's runtime behavior and import requirements.

    A ``client_patch`` integration is activated by the
    ``install_auto_load_patch`` function of the adapter module with its name.
    """

    name: str
    required_modules: tuple[str, ...]
    kind: IntegrationKind
    adapter_module: str | None = None

    @property
    def installer(self) -> str | None:
        if self.kind != "client_patch":
            return None
        return f"{ADAPTER_PACKAGE}.{self.adapter_module or self.name}:install_auto_load_patch"


INTEGRATION_SPECS: tuple[IntegrationSpec, ...] = (
    IntegrationSpec("anthropic", ("anthropic",), "client_patch"),
    IntegrationSpec("apple_fm", ("apple_fm_sdk",), "client_patch"),
    IntegrationSpec("llama_cpp", ("llama_cpp",), "client_patch", adapter_module="gguf"),
    IntegrationSpec("onnx", ("onnxruntime",), "client_patch"),
    IntegrationSpec("openai", ("openai",), "client_patch"),
    IntegrationSpec("timm", ("timm",), "client_patch"),
    IntegrationSpec("torch", ("torch",), "noop"),
    IntegrationSpec("keras", ("keras",), "noop"),
    IntegrationSpec("tensorflow", ("tensorflow",), "client_patch"),
    IntegrationSpec("ultralytics", ("ultralytics",), "client_patch"),
    IntegrationSpec("transformers", ("transformers",), "client_patch"),
    IntegrationSpec("faster_whisper", ("faster_whisper",), "client_patch"),
    IntegrationSpec("mlx_whisper", ("mlx_whisper",), "client_patch"),
    IntegrationSpec("mlx", ("mlx_lm",), "client_patch"),
)

INTEGRATIONS_BY_NAME: dict[str, IntegrationSpec] = {
    spec.name: spec for spec in INTEGRATION_SPECS
}


# Public names describe runtimes; retain the pre-0.3 name as an input alias.
INTEGRATION_ALIASES = {"gguf": "llama_cpp"}
INTEGRATIONS_BY_NAME.update(
    {alias: INTEGRATIONS_BY_NAME[name] for alias, name in INTEGRATION_ALIASES.items()}
)


def canonical_integration(name: str) -> str:
    return INTEGRATION_ALIASES.get(name, name)


def supported_integrations() -> set[str]:
    """Return all supported integration names."""
    return {spec.name for spec in INTEGRATION_SPECS}


def noop_integrations() -> set[str]:
    """Return integrations that intentionally perform no global patching."""
    return {name for name, spec in INTEGRATIONS_BY_NAME.items() if spec.kind == "noop"}


@dataclass(frozen=True)
class ExtractorSpec:
    """An extractor, and cheap checks for when it is worth importing.

    ``could_handle`` only decides whether to import the adapter; its
    ``can_handle`` still makes the final call.
    """

    module: str  # adapter module in wildedge.integrations
    class_name: str  # extractor class in that module
    # Packages whose classes the extractor handles, matched anywhere in the
    # object's class hierarchy so application subclasses count too.
    packages: tuple[str, ...] = ()
    # Exact class names, for extractors whose can_handle checks the name.
    class_names: tuple[str, ...] = ()
    # Attribute the SDK sets on objects it has already described.
    marker_attr: str | None = None

    def could_handle(self, obj: object) -> bool:
        if f"{ADAPTER_PACKAGE}.{self.module}" in sys.modules:
            return True  # already imported, so asking it costs nothing
        if type(obj).__name__ in self.class_names:
            return True
        if self.marker_attr and getattr(obj, self.marker_attr, None) is not None:
            return True
        return any(
            cls.__module__.partition(".")[0] in self.packages
            for cls in type(obj).__mro__
        )

    def load(self):
        return load_extractor(f"{ADAPTER_PACKAGE}.{self.module}:{self.class_name}")


@cache
def load_extractor(target: str):
    return resolve(target)()


# Order matters: the first extractor that handles a model wins, so wrappers
# and specialized formats come before the generic tensor frameworks.
EXTRACTOR_SPECS: tuple[ExtractorSpec, ...] = (
    ExtractorSpec("onnx", "OnnxExtractor", ("onnxruntime",)),
    ExtractorSpec("gguf", "GgufExtractor", class_names=("Llama",)),
    ExtractorSpec(
        "openai",
        "OpenAIExtractor",
        class_names=(
            "OpenAI",
            "AsyncOpenAI",
            "Completions",
            "AsyncCompletions",
            "Responses",
            "AsyncResponses",
            "Embeddings",
            "AsyncEmbeddings",
        ),
    ),
    ExtractorSpec(
        "anthropic",
        "AnthropicExtractor",
        class_names=("Anthropic", "AsyncAnthropic", "Messages", "AsyncMessages"),
    ),
    ExtractorSpec("apple_fm", "AppleFmExtractor", ("apple_fm_sdk",)),
    ExtractorSpec("ultralytics", "UltralyticsExtractor", class_names=("YOLO",)),
    ExtractorSpec("transformers", "TransformersExtractor", ("transformers",)),
    # Before MLX, which would otherwise claim MLX Whisper models.
    ExtractorSpec(
        "whisper",
        "WhisperExtractor",
        class_names=("WhisperDescriptor",),
        marker_attr="__wildedge_whisper_descriptor__",
    ),
    ExtractorSpec("mlx", "MlxExtractor", ("mlx",)),
    ExtractorSpec("pytorch", "PytorchExtractor", ("torch",)),
    ExtractorSpec(
        "tensorflow", "TensorflowExtractor", ("tensorflow", "keras", "tf_keras")
    ),
    ExtractorSpec("keras", "KerasExtractor", ("keras", "tf_keras")),
)


def find_extractor(obj: object):
    """The first extractor that handles ``obj``, or None."""
    for spec in EXTRACTOR_SPECS:
        if spec.could_handle(obj):
            extractor = spec.load()
            if extractor.can_handle(obj):
                return extractor
    return None
