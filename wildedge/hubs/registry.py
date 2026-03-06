"""Hub/repository tracker registry.

Separate from the framework integration registry (``wildedge.integrations.registry``)
because hubs answer "where did the model come from?" while integrations answer
"how does the model run?".  These are orthogonal concerns with different
activation semantics and different lifecycle hooks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HubSpec:
    """Describes one hub tracker's identity and import requirements."""

    name: str
    required_modules: tuple[str, ...]


HUB_SPECS: tuple[HubSpec, ...] = (
    HubSpec("huggingface", ("huggingface_hub",)),
    HubSpec("torchhub", ("torch",)),
)

HUBS_BY_NAME: dict[str, HubSpec] = {spec.name: spec for spec in HUB_SPECS}


def supported_hubs() -> set[str]:
    """Return all supported hub tracker names."""
    return set(HUBS_BY_NAME.keys())
