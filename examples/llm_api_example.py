# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""Track LLM calls made over plain HTTP with wildedge.llm_api().

No openai or anthropic client library involved: requests go through stdlib
urllib against OpenRouter's OpenAI-compatible endpoint, and llm_api() records
the same inference events the integrations emit (tokens, TTFT, stop reason),
correlated into the surrounding trace and spans.

Run with: uv run llm_api_example.py
Requires: OPENROUTER_API_KEY environment variable. Set WILDEDGE_DSN to send events.
"""

import json
import os
import urllib.request
import uuid

import wildedge

wildedge.init(app_version="1.0.0")  # uses WILDEDGE_DSN if set; otherwise no-op

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"


def chat(prompt: str) -> dict:
    request = urllib.request.Request(
        URL,
        data=json.dumps(
            {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}
        ).encode(),
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


PROMPTS = [
    "What is on-device AI in one sentence?",
    "Name three edge inference runtimes.",
]

with wildedge.trace(agent_id="llm-api-example", run_id=str(uuid.uuid4())):
    for step, prompt in enumerate(PROMPTS):
        with wildedge.span(kind="agent_step", name="ask", step_index=step):
            with wildedge.llm_api(
                model=MODEL, provider="openrouter", prompt=prompt
            ) as call:
                data = chat(prompt)
                call.response(data)
        print(f"Q: {prompt}\nA: {data['choices'][0]['message']['content']}\n")

wildedge.flush()
print("Done. Events flushed to WildEdge.")
