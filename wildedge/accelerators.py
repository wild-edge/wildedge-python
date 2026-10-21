"""Accelerator names reported in model load events and hardware context.

Integrations report the most specific name they know. PyTorch device types
(for example ``xpu``) pass through as they are, so this list is not closed.
"""

CPU = "cpu"
CUDA = "cuda"  # NVIDIA GPU
ROCM = "rocm"  # AMD GPU
MPS = "mps"  # Apple GPU (Metal)
COREML = "coreml"  # Apple Core ML, which may use the GPU or Neural Engine
DIRECTML = "directml"  # Windows DirectML
OPENVINO = "openvino"  # Intel OpenVINO
