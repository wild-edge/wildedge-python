# Apple Foundation Models

The `apple_fm` integration tracks the on-device model behind Apple
Intelligence, called through Apple's Python SDK
([apple-fm-sdk](https://github.com/apple/python-apple-fm-sdk)). It needs macOS
26 or later on Apple silicon with Apple Intelligence turned on, plus Xcode to
install the SDK.

Initialize the SDK before creating sessions:

```python
import asyncio

import apple_fm_sdk as fm
import wildedge

client = wildedge.init(integrations=["apple_fm"])


async def main():
    session = fm.LanguageModelSession(instructions="Answer in one sentence.")
    print(await session.respond("How does rain form?"))
    # Each snapshot is the full text so far, not just the new part.
    text = ""
    async for snapshot in session.stream_response("Name three clouds."):
        text = snapshot
    print(text)


asyncio.run(main())
client.close(timeout=10)
```

## What is recorded

Each `respond()` or `stream_response()` call records one inference event with
its duration, the prompt's character and word counts, and the
`GenerationOptions` you passed (temperature, maximum response tokens, and
sampling top-k, threshold and seed). Streams also record time to first token.

Sessions share one model per use case: `apple-foundation-model` for the
general model, and `apple-foundation-model-content-tagging` for
`SystemLanguageModelUseCase.CONTENT_TAGGING`. The model version is the macOS
version, since the model ships with the OS.

Failed calls record an error event named after the SDK's error, such as
`guardrail_violation`, `exceeded_context_window_size`, `assets_unavailable` or
`refusal`. A model that isn't ready because Apple Intelligence is off fails
this way too.

## Caveats

- Token counts and tokens per second are empty. The SDK doesn't report token
  usage, and counting tokens with `SystemLanguageModel.token_count()` is an
  extra call to the model, so WildEdge doesn't make it for you.
- Prompts and responses are not uploaded.

## `fm serve`

On macOS 27, `fm serve` exposes the same model through a local
OpenAI-compatible server. Calls made to it with the `openai` client are
tracked by the `openai` integration. Name the provider, because a local host is
otherwise recorded as `127.0.0.1` (see [Providers](providers.md#other-hosts)).
