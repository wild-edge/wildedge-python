"""Quantization labels reported in model metadata.

Common weight precisions use the names below. Format-specific schemes pass
through as their runtime names them, such as GGUF's ``q4_k_m`` or
bitsandbytes' ``nf4``, so this list is not closed.
"""

F32 = "f32"
F16 = "f16"
BF16 = "bf16"
INT8 = "int8"
QUANTIZED = "quantized"  # quantized, with an unknown bit width


def int_bits(bits: int) -> str:
    """Label for integer weights of the given width, such as ``int4``."""
    return f"int{int(bits)}"
