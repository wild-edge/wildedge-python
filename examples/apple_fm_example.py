# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "apple-fm-sdk"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""Apple Foundation Models integration example.

WildEdge patches apple_fm_sdk.LanguageModelSession at instrumentation time, so
every respond() and stream_response() call on Apple's on-device model is
tracked.

Run with: uv run apple_fm_example.py
Requires: macOS 26+ on Apple silicon with Apple Intelligence on, and Xcode to
install apple-fm-sdk. Set WILDEDGE_DSN to send events.
"""

import asyncio

import apple_fm_sdk as fm

import wildedge

client = wildedge.init(
    app_version="1.0.0",  # uses WILDEDGE_DSN if set; otherwise no-op
    integrations="apple_fm",
)


async def main() -> None:
    model = fm.SystemLanguageModel()
    available, reason = model.is_available()
    if not available:
        raise SystemExit(f"Apple's on-device model is unavailable: {reason}")

    session = fm.LanguageModelSession(
        instructions="Answer in one sentence.", model=model
    )
    print(await session.respond("Explain transformer attention."))

    # Each snapshot is the full text so far, not just the new part.
    text = ""
    async for snapshot in session.stream_response("Write a haiku about edge AI."):
        text = snapshot
    print(text)


try:
    asyncio.run(main())
finally:
    client.close(timeout=10)
