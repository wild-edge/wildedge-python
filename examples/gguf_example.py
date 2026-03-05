# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge", "llama-cpp-python", "huggingface_hub"]
#
# [tool.uv.sources]
# wildedge = { path = "..", editable = true }
# ///
"""GGUF / llama.cpp integration example — run with: uv run gguf_example.py"""

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

import wildedge

client = wildedge.WildEdge(
    app_version="1.0.0",  # set WILDEDGE_DSN env var
)
client.instrument("huggingface")
client.instrument("gguf")

model_path = hf_hub_download(
    "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
)
llm = Llama(model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)

prompts = [
    "Explain what a transformer architecture is in one sentence.",
    "What is the capital of France?",
    "Write a haiku about on-device AI.",
]

for prompt in prompts:
    result = llm(prompt, max_tokens=128, temperature=0.7)
    text = result["choices"][0]["text"].strip()
    print(f"Q: {prompt}\nA: {text}\n")
