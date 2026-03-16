# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "llama-cpp-python", "huggingface_hub"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""Gemma 2 GGUF: fully manual integration, no auto-instrumentation.

Shows explicit download / load / inference / error tracking without
client.instrument() or any automatic hooks.

Run with: uv run gguf_gemma_manual_example.py
"""

import os

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

import wildedge
from wildedge import GenerationOutputMeta, TextInputMeta, capture_hardware
from wildedge.timing import Timer

REPO = "bartowski/gemma-2-2b-it-GGUF"
FILE = "gemma-2-2b-it-Q4_K_M.gguf"

client = wildedge.WildEdge(
    app_version="1.0.0",  # set WILDEDGE_DSN env var
)

# --- Download ---
with Timer() as t:
    model_path = hf_hub_download(REPO, FILE)
file_size = os.path.getsize(model_path)
cache_hit = t.elapsed_ms < 500  # cached downloads are near-instant

# --- Load ---
with Timer() as t:
    llm = Llama(model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
load_ms = t.elapsed_ms

# All metadata supplied explicitly. No extractor runs, no hooks installed.
handle = client.register_model(
    llm,
    model_id="gemma-2-2b-it-q4",
    family="gemma",
    version="2",
    quantization="q4_k_m",
    auto_instrument=False,
)

handle.track_download(
    source_url=f"hf://{REPO}/{FILE}",
    source_type="huggingface",
    file_size_bytes=file_size,
    downloaded_bytes=0 if cache_hit else file_size,
    duration_ms=t.elapsed_ms,
    network_type="unknown",
    resumed=False,
    cache_hit=cache_hit,
    success=True,
)

handle.track_load(
    duration_ms=load_ms,
    memory_bytes=file_size,
    gpu_layers=-1,
    context_length=2048,
)

# --- Inference ---
prompts = [
    "Explain what a transformer architecture is in one sentence.",
    "What is the capital of France?",
    "Write a haiku about on-device AI.",
]

for prompt in prompts:
    try:
        with Timer() as t:
            result = llm(prompt, max_tokens=128, temperature=0.7)

        usage = result.get("usage", {})
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
        tps = (
            round(tokens_out / t.elapsed_ms * 1000, 1)
            if tokens_out and t.elapsed_ms > 0
            else None
        )

        handle.track_inference(
            duration_ms=t.elapsed_ms,
            hardware=capture_hardware(),
            input_modality="text",
            output_modality="text",
            success=True,
            input_meta=TextInputMeta(
                char_count=len(prompt),
                word_count=len(prompt.split()),
                token_count=tokens_in,
            ),
            output_meta=GenerationOutputMeta(
                task="generation",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tokens_per_second=tps,
            ),
        )

        text = result["choices"][0]["text"].strip()
        print(f"Q: {prompt}\nA: {text}\n")

    except Exception as exc:
        handle.track_error(error_code="UNKNOWN", error_message=str(exc)[:200])
        raise
