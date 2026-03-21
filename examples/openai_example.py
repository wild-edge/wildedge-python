# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "openai"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""OpenAI integration example.

WildEdge patches openai.OpenAI (and AsyncOpenAI) at instrumentation time, so
inference tracking happens automatically for every chat.completions.create call.

Run with: uv run openai_example.py
Requires: WILDEDGE_DSN and OPENAI_API_KEY environment variables.
"""

from openai import OpenAI

import wildedge

client = wildedge.WildEdge(
    app_version="1.0.0"
)  # uses WILDEDGE_DSN if set; otherwise no-op
client.instrument("openai")

openai_client = OpenAI()  # set OPENAI_API_KEY env var or pass api_key= explicitly

prompts = [
    "Explain transformer attention in one sentence.",
    "What is the capital of Japan?",
    "Write a haiku about edge AI.",
]

for prompt in prompts:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=256,
    )
    print(f"Q: {prompt}\nA: {response.choices[0].message.content}\n")

client.flush()
print("Done. Events flushed to WildEdge.")
