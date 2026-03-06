"""Model hub and repository trackers.

Hub trackers record *where* a model came from (download provenance, cache hits,
bandwidth) independently of *how* the model runs (handled by framework
integrations in ``wildedge.integrations``).

Supported hubs
--------------
``huggingface``
    Patches ``huggingface_hub.hf_hub_download`` and ``snapshot_download``.
    Also provides filesystem-diff support for timm models that download from
    HuggingFace Hub implicitly inside ``create_model()``.
    Requires: ``huggingface-hub``.

``torchhub``
    Patches ``torch.hub.load`` and scans the torch hub cache directory for
    files downloaded via ``torch.hub.download_url_to_file`` (used by older
    timm model families).  Emits ``source_type='torchhub'`` records, fixing
    the incorrect ``source_type='url'`` labelling in the previous
    implementation.
    Requires: ``torch``.

Activation
----------
Hub trackers are activated via ``client.instrument(name)``, the same API used
for framework integrations.  Internally the client routes hub names to
``_activate_hub()`` rather than the framework patch-installer dispatch::

    client.instrument("onnx")         # framework: inference tracking
    client.instrument("huggingface")  # hub: download provenance tracking
    client.instrument("torchhub")     # hub: torch.hub.load tracking
"""

from wildedge.hubs.base import BaseHubTracker
from wildedge.hubs.huggingface import HuggingFaceHubTracker
from wildedge.hubs.registry import HUB_SPECS, HUBS_BY_NAME, HubSpec, supported_hubs
from wildedge.hubs.torchhub import TorchHubTracker

__all__ = [
    "BaseHubTracker",
    "HuggingFaceHubTracker",
    "TorchHubTracker",
    "HubSpec",
    "HUB_SPECS",
    "HUBS_BY_NAME",
    "supported_hubs",
]
