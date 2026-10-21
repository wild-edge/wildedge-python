"""timm integration: times ``timm.create_model`` and records its downloads.

The instrumentation lives with the PyTorch adapter, since timm models are
PyTorch modules. This module is the entry point the integration registry names.
"""

from wildedge.integrations.pytorch import PytorchExtractor

install_auto_load_patch = PytorchExtractor.install_timm_patch
