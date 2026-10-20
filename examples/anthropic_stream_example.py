# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "anthropic>=0.25.0"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""Two small billable native streaming calls, sync text and async final message.

Requires ANTHROPIC_API_KEY and WILDEDGE_DSN; optionally ANTHROPIC_MODEL.
From the SDK root: uv run examples/anthropic_stream_example.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

from anthropic import Anthropic, AsyncAnthropic

import wildedge

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
MESSAGES = [{"role": "user", "content": "Reply with exactly: Hello from WildEdge"}]


def run_sync() -> None:
    with (
        Anthropic() as client,
        wildedge.span(kind="agent_step", name="anthropic-native-text"),
    ):
        with client.messages.stream(
            model=MODEL, max_tokens=64, messages=MESSAGES
        ) as stream:
            text = "".join(stream.text_stream)
            result = stream.get_final_message()
        if not text or not result.usage:
            raise RuntimeError("Missing output or usage")
        print(
            f"Sync text: input={result.usage.input_tokens}, output={result.usage.output_tokens}"
        )


async def run_async() -> None:
    async with AsyncAnthropic() as client:
        with wildedge.span(kind="agent_step", name="anthropic-native-final"):
            async with client.messages.stream(
                model=MODEL, max_tokens=64, messages=MESSAGES
            ) as stream:
                result = await stream.get_final_message()
        if not result.content or not result.usage:
            raise RuntimeError("Missing output or usage")
        print(
            f"Async final: input={result.usage.input_tokens}, output={result.usage.output_tokens}"
        )


def main() -> None:
    missing = [
        name
        for name in ("ANTHROPIC_API_KEY", "WILDEDGE_DSN")
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit("Set required environment variables: " + ", ".join(missing))
    run_id = "anthropic-native-e2e-" + uuid.uuid4().hex[:12]
    print(f"Run ID: {run_id}")
    sdk = wildedge.init(integrations=["anthropic"], app_version="anthropic-native-e2e")
    try:
        with wildedge.trace(run_id=run_id):
            run_sync()
            asyncio.run(run_async())
    finally:
        sdk.close(timeout=15)
    print("Provider checks passed. Verify two inferences and two spans in WildEdge.")


if __name__ == "__main__":
    main()
