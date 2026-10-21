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
    tracker_class: str  # tracker class in the wildedge.hubs module with this name

    @property
    def tracker(self) -> str:
        """Import path of the tracker class, loaded only when the hub is used."""
        return f"wildedge.hubs.{self.name}:{self.tracker_class}"


HUB_SPECS: tuple[HubSpec, ...] = (
    HubSpec("huggingface", ("huggingface_hub",), "HuggingFaceHubTracker"),
    HubSpec("torchhub", ("torch",), "TorchHubTracker"),
)

HUBS_BY_NAME: dict[str, HubSpec] = {spec.name: spec for spec in HUB_SPECS}


def supported_hubs() -> set[str]:
    """Return all supported hub tracker names."""
    return set(HUBS_BY_NAME.keys())
