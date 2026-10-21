# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "llama-cpp-python", "huggingface_hub"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""GGUF / llama.cpp integration example. Run with: uv run gguf_example.py

The llama_cpp integration tracks every llama-cpp-python entry point, once per
call: create_chat_completion (streamed or not), create_completion and
llm(...). Each inference event carries token counts, tokens per second,
the stop reason and the sampling settings; chat calls also record
prompt_type="chat" and the turn index.
"""

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

import wildedge

wildedge.init(
    app_version="1.0.0",  # uses WILDEDGE_DSN if set; otherwise no-op
    integrations="llama_cpp",
    hubs=["huggingface"],
)

model_path = hf_hub_download(
    "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
)
llm = Llama(model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)

# A short multi-turn chat, streamed. Each turn is one inference event.
messages = [{"role": "system", "content": "You are a concise assistant."}]
for question in [
    "What is a transformer architecture, in one sentence?",
    "And why does it suit on-device AI?",
    "Now say that as a haiku.",
]:
    messages.append({"role": "user", "content": question})
    print(f"Q: {question}\nA: ", end="", flush=True)
    reply = ""
    for chunk in llm.create_chat_completion(
        messages=messages, max_tokens=128, temperature=0.7, top_p=0.9, stream=True
    ):
        token = chunk["choices"][0]["delta"].get("content") or ""
        reply += token
        print(token, end="", flush=True)
    print("\n")
    messages.append({"role": "assistant", "content": reply})

# Plain text completion, not streamed.
result = llm.create_completion(
    "The capital of France is", max_tokens=8, temperature=0.0, stop=["."]
)
print(f"Completion: {result['choices'][0]['text'].strip()}")
