"""Framework integration registry.

Contains only ML *framework* integrations (inference tracking, load/unload
timing).  Model hub and repository trackers (download provenance) live in
``wildedge.hubs.registry`` — they are orthogonal concerns with different
activation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IntegrationKind = Literal["noop", "client_patch"]


@dataclass(frozen=True)
class IntegrationSpec:
    """Describes one integration's runtime behavior and import requirements."""

    name: str
    required_modules: tuple[str, ...]
    kind: IntegrationKind


INTEGRATION_SPECS: tuple[IntegrationSpec, ...] = (
    IntegrationSpec("gguf", ("llama_cpp",), "client_patch"),
    IntegrationSpec("onnx", ("onnxruntime",), "client_patch"),
    IntegrationSpec("timm", ("timm",), "client_patch"),
    IntegrationSpec("torch", ("torch",), "noop"),
    IntegrationSpec("keras", ("keras",), "noop"),
    IntegrationSpec("tensorflow", ("tensorflow",), "client_patch"),
    IntegrationSpec("ultralytics", ("ultralytics",), "client_patch"),
    IntegrationSpec("transformers", ("transformers",), "client_patch"),
)

INTEGRATIONS_BY_NAME: dict[str, IntegrationSpec] = {
    spec.name: spec for spec in INTEGRATION_SPECS
}


def supported_integrations() -> set[str]:
    """Return all supported integration names."""
    return set(INTEGRATIONS_BY_NAME.keys())


def noop_integrations() -> set[str]:
    """Return integrations that intentionally perform no global patching."""
    return {name for name, spec in INTEGRATIONS_BY_NAME.items() if spec.kind == "noop"}
