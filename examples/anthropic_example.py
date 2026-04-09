# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "anthropic"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""Anthropic integration example.

WildEdge patches anthropic.Anthropic (and AsyncAnthropic) at instrumentation
time, so inference tracking happens automatically for every messages.create call.

Run with: uv run anthropic_example.py
Requires: ANTHROPIC_API_KEY environment variable. Set WILDEDGE_DSN to send events.
"""

import anthropic

import wildedge

client = wildedge.init(
    app_version="1.0.0",  # uses WILDEDGE_DSN if set; otherwise no-op
    integrations="anthropic",
)

anthropic_client = anthropic.Anthropic()  # set ANTHROPIC_API_KEY or pass api_key=

prompts = [
    "Explain transformer attention in one sentence.",
    "What is the capital of Japan?",
    "Write a haiku about edge AI.",
]

for prompt in prompts:
    stream = anthropic_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    print(f"Q: {prompt}\nA: ", end="", flush=True)
    for event in stream:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            print(event.delta.text, end="", flush=True)
    print("\n")

client.flush()
print("Done. Events flushed to WildEdge.")
