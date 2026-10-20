# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "openai>=1.30.0"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""Run two small billable embedding calls; requires OPENAI_API_KEY and WILDEDGE_DSN.

From the SDK root: uv run examples/openai_embeddings_example.py
Optionally set OPENAI_EMBEDDING_MODEL. Find the printed run ID in WildEdge.
"""

from __future__ import annotations

import asyncio
import os
import uuid

from openai import AsyncOpenAI, OpenAI

import wildedge

MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
INPUT = ["Hello from WildEdge", "Embedding instrumentation check"]


def check_result(result: object) -> None:
    if len(result.data) != len(INPUT) or not result.data[0].embedding:
        raise RuntimeError("Expected two nonempty embeddings")
    if result.usage is None:
        raise RuntimeError("Provider returned no usage")
    print(
        f"Completed: batch={len(result.data)}, dimensions={len(result.data[0].embedding)}, input_tokens={result.usage.prompt_tokens}"
    )


def run_sync() -> None:
    with OpenAI() as client, wildedge.span(kind="agent_step", name="embeddings-sync"):
        check_result(
            client.embeddings.create(model=MODEL, input=INPUT, encoding_format="float")
        )


async def run_async() -> None:
    async with AsyncOpenAI() as client:
        with wildedge.span(kind="agent_step", name="embeddings-async"):
            check_result(
                await client.embeddings.create(
                    model=MODEL, input=INPUT, encoding_format="float"
                )
            )


def main() -> None:
    missing = [
        key for key in ("OPENAI_API_KEY", "WILDEDGE_DSN") if not os.environ.get(key)
    ]
    if missing:
        raise SystemExit("Set required environment variables: " + ", ".join(missing))
    run_id = "embeddings-e2e-" + uuid.uuid4().hex[:12]
    print(f"Run ID: {run_id}")
    sdk = wildedge.init(integrations=["openai"], app_version="embeddings-e2e")
    try:
        with wildedge.trace(run_id=run_id):
            run_sync()
            asyncio.run(run_async())
    finally:
        sdk.close(timeout=15)
    print(
        "Provider checks passed. Verify two inference events and two spans in WildEdge."
    )


if __name__ == "__main__":
    main()
