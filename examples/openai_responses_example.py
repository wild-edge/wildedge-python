# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "openai>=1.66.0"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""Exercise all four Responses paths against OpenAI and WildEdge.

From the SDK root:
    uv run examples/openai_responses_example.py

Requires OPENAI_API_KEY and WILDEDGE_DSN. Optionally set OPENAI_MODEL.
Makes four small, billable API calls. Find the printed run ID in WildEdge.
"""

from __future__ import annotations

import asyncio
import os
import uuid

from openai import AsyncOpenAI, OpenAI

import wildedge

PROMPT = "Reply with exactly: Hello from Responses"
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def check_response(result: object, label: str) -> None:
    if result.status != "completed" or not result.output_text:
        raise RuntimeError(f"{label}: response did not complete with text")
    if result.usage is None:
        raise RuntimeError(f"{label}: provider returned no usage")
    print(
        f"{label}: completed; input={result.usage.input_tokens}, output={result.usage.output_tokens}"
    )


def run_sync() -> None:
    with OpenAI() as client:
        with wildedge.span(kind="agent_step", name="responses-sync"):
            result = client.responses.create(
                model=MODEL, input=PROMPT, max_output_tokens=64
            )
            check_response(result, "sync")
        with wildedge.span(kind="agent_step", name="responses-sync-stream"):
            terminal = None
            with client.responses.create(
                model=MODEL, input=PROMPT, max_output_tokens=64, stream=True
            ) as stream:
                for event in stream:
                    if event.type == "response.completed":
                        terminal = event.response
            if terminal is None:
                raise RuntimeError("sync stream: missing completed response")
            check_response(terminal, "sync stream")


async def run_async() -> None:
    async with AsyncOpenAI() as client:
        with wildedge.span(kind="agent_step", name="responses-async"):
            result = await client.responses.create(
                model=MODEL, input=PROMPT, max_output_tokens=64
            )
            check_response(result, "async")
        with wildedge.span(kind="agent_step", name="responses-async-stream"):
            terminal = None
            async with await client.responses.create(
                model=MODEL, input=PROMPT, max_output_tokens=64, stream=True
            ) as stream:
                async for event in stream:
                    if event.type == "response.completed":
                        terminal = event.response
            if terminal is None:
                raise RuntimeError("async stream: missing completed response")
            check_response(terminal, "async stream")


def main() -> None:
    missing = [
        key for key in ("OPENAI_API_KEY", "WILDEDGE_DSN") if not os.environ.get(key)
    ]
    if missing:
        raise SystemExit("Set required environment variables: " + ", ".join(missing))
    run_id = f"responses-e2e-{uuid.uuid4().hex[:12]}"
    print(f"Run ID: {run_id}")
    sdk = wildedge.init(integrations=["openai"], app_version="responses-e2e")
    try:
        with wildedge.trace(run_id=run_id):
            run_sync()
            asyncio.run(run_async())
    finally:
        sdk.close(timeout=15)
    print("Provider checks passed. SDK shutdown finished; verify delivery in WildEdge.")
    print("Expect four inference events under this run, one beneath each named span.")


if __name__ == "__main__":
    main()
