# Streaming lifecycle and coverage

The shared wrappers track OpenAI Chat Completions and Responses
`create(stream=True)`, Anthropic `messages.create(stream=True)` (sync and async),
and llama-cpp-python/GGUF completion iterators (sync).

| Outcome | Telemetry |
|---|---|
| Iterator exhausted | One inference event; remote APIs require their completion marker |
| Explicit `close()` / `aclose()` or early context exit | One unsuccessful inference with `STREAM_CLOSED`, unless already finalized |
| Async cancellation while awaiting a chunk or exiting its context | One unsuccessful inference with `STREAM_CANCELLED`; cancellation is re-raised |
| Provider/iterator exception | One error event using existing error classification; original exception is re-raised |
| Remote API EOF without completion marker | One unsuccessful inference with `STREAM_INCOMPLETE` |

Repeated exhaustion and closing after finalization do not emit another outcome.
Partial usage is retained when supplied by the provider, but missing totals are
not estimated. Responses interrupted before a terminal response has no final
usage; first-output timing is retained when available. Telemetry callback failures
are logged at debug level and do not break application iteration.

TTFT measures from the original call until the first nonempty output delta,
ignoring role-only chunks, usage chunks, and message/response lifecycle events.
Output can be text, refusal, tool arguments, or supported audio/reasoning deltas;
it need not be user-visible text. No output means TTFT is omitted. Callbacks use
the trace context captured when the stream wrapper was created, even when the
application consumes it later under another context.

Use a context manager or explicitly close the stream when stopping early:

```python
with client.chat.completions.create(model=model, messages=messages, stream=True) as stream:
    for chunk in stream:
        if enough_output(chunk):
            break
```

Async streams support `async with`, `await stream.close()`, and
`await stream.aclose()`. A bare `break` without closing, or simply abandoning an
unconsumed stream, does not provide an observable end of the operation. There is
no garbage-collection finalizer. Keep stream consumption within the lifetime of
the SDK client and do not consume one iterator concurrently from multiple tasks.

## Audit findings outside these wrappers

- Anthropic's `messages.stream()` helper calls its transport directly and bypasses
  the currently instrumented `messages.create()`. Use `create(stream=True)` for
  the coverage above. Adding native helper support needs its own instrumentation
  for `text_stream` and `get_final_message()`.
- Provider convenience properties/methods that consume an underlying stream
  directly can bypass iteration on the wrapper. They are not covered by the
  lifecycle contract above; use the wrapped event iterator.
- MLX instruments `mlx_lm.generate()` and model calls, not
  `mlx_lm.stream_generate()`. Streaming generation requires a dedicated adapter
  and suppression of nested token-loop model calls.
- Transformers instruments pipeline/model calls; user-supplied streamers and
  background generation threads are not wrapped stream iterators.
- `wildedge.llm_api()` is a manual context manager: its lifetime is controlled by
  the caller, and the caller must supply usage and first-output marks.

The audit does not claim coverage for streaming APIs that are not instrumented.
