# Providers

Any OpenAI- or Anthropic-compatible endpoint works: set `base_url` on the
client, or pass `provider=` to [`wildedge.llm_api()`](llm_api.md). The host is
mapped to a `model_source` name on every event.

```python
OpenAI(base_url="https://api.fireworks.ai/inference/v1")
```

## Direct HTTP calls

For calls made with httpx, requests, or urllib instead of the OpenAI or
Anthropic client libraries, use [`wildedge.llm_api()`](llm_api.md).
See the [direct HTTP example](../examples/llm_api_example.py) for usage.

## Recognized providers

| Provider | Base URL | `model_source` |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `openai` |
| Anthropic | `https://api.anthropic.com` | `anthropic` |
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/v1` | `azure-openai` |
| OpenRouter | `https://openrouter.ai/api/v1` | `openrouter` |
| Fireworks AI | `https://api.fireworks.ai/inference/v1` | `fireworks` |
| Together AI | `https://api.together.xyz/v1` | `together` |
| Baseten (Model APIs) | `https://inference.baseten.co/v1` | `baseten` |
| Baseten (dedicated) | `https://model-<id>.api.baseten.co/environments/production/sync/v1` | `baseten` |
| xAI | `https://api.x.ai/v1` | `xai` |
| Mistral | `https://api.mistral.ai/v1` | `mistral` |
| Groq | `https://api.groq.com/openai/v1` | `groq` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek` |
| Cerebras | `https://api.cerebras.ai/v1` | `cerebras` |
| Perplexity | `https://api.perplexity.ai` | `perplexity` |
| NVIDIA NIM | `https://integrate.api.nvidia.com/v1` | `nvidia` |
| Hugging Face | `https://router.huggingface.co/v1` | `huggingface` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `google` |

Azure OpenAI and dedicated Baseten deployments match by domain suffix, so every
resource and deployment subdomain resolves to the same name.

## Other hosts

Unrecognized hosts are recorded verbatim, so self-hosted runtimes (vLLM, Ollama,
LM Studio, llama.cpp) all collapse onto `localhost` or `127.0.0.1`. Name them:

```python
wildedge.llm_api(model="gemma-7b", provider="ollama")
wildedge.register_model(client, source="ollama", model_id="gemma-7b")
```

The same applies to `fm serve`, Apple's local server for its on-device model on
macOS 27. To track that model without a server, see [Apple Foundation Models](apple_fm.md).

## Caveats

- `tokens_in` / `tokens_out` are populated by every provider above.
- `cached_input_tokens` and `reasoning_tokens_out` are OpenAI extensions. Most
  providers omit them; they record as `None`, not `0`.
- Streamed responses carry no usage unless you pass
  `stream_options={"include_usage": True}`. Timing and TTFT are unaffected.


## Responses API

The `openai` integration instruments `client.responses.create()` and
`await client.responses.create()`, with or without `stream=True`. Enable it
before constructing the OpenAI client, as with Chat Completions.

```python
import wildedge
from openai import OpenAI

wildedge.init(integrations=["openai"])
client = OpenAI()
response = client.responses.create(model="gpt-4o-mini", input="Say hello")
print(response.output_text)
wildedge.flush()
```

Usage comes from Responses' `input_tokens` / `output_tokens`, including cached
input and reasoning tokens when provided. The stop reason is the response
status, or the incomplete reason such as `max_output_tokens`. Failed responses
are recorded as unsuccessful. No prompt or response text is uploaded by this
adapter. String input and simple user messages provide text input metadata;
input iterators are passed through without being consumed by instrumentation.

See [Streaming lifecycle](streaming.md) for completion, closure, cancellation,
and coverage limits.

For streams, consume the iterator to exhaustion to record the terminal response
and its usage. Time to first output ignores lifecycle events and measures the
first nonempty text, refusal, function argument, or audio delta. Streams without
a terminal response record an unsuccessful inference. Explicit early closure and
async cancellation are tracked; background response polling and individual tool
spans remain outside coverage. API exceptions retain their original behavior and use
the same error tracking as Chat Completions.

OpenAI-compatible hosts must implement the Responses endpoint to use this path.
Older OpenAI SDK versions without a `responses` resource continue to support
Chat Completions.


### End-to-end check

From the SDK root, with `OPENAI_API_KEY` and `WILDEDGE_DSN` set:

```bash
uv run examples/openai_responses_example.py
```

This uses the local SDK checkout and makes four small, billable requests:
sync, sync streaming, async, and async streaming. Set `OPENAI_MODEL` to override
the default `gpt-4o-mini`. The script checks completion and usage, prints a
unique `responses-e2e-...` run ID, and closes the SDK with a 15-second deadline.
In WildEdge, locate that run and verify four inference events beneath the four
named spans, with input/output token usage and first-output timing on the two
streams. Provider checks alone do not prove telemetry delivery; confirm these
events in the dashboard.


## Embeddings

The `openai` integration also tracks sync and async `client.embeddings.create()`
calls. Enable instrumentation before constructing the client. OpenAI-compatible
providers work through the same integration if they implement the embeddings
endpoint.

Records latency, resolved model, input token usage, batch size, and dimensions.
Text, batches of strings, token ID arrays, and batches of token ID arrays are
supported. Arbitrary iterables pass through without instrumentation consuming
them; their batch size is omitted. Dimensions come from the returned vector
length, or float32 byte length for explicit base64 output. Usage comes from the
provider, not a local tokenizer. Both float and base64 vectors remain in the
application response and are never included in telemetry. Raw inputs are also
not uploaded by this adapter.

Input usage is recorded in `input_meta.token_count` and `output_meta.tokens_in`
so existing analytics can query it. Embeddings do not emit generated-token
counts, throughput, or TTFT. API exceptions use the existing error tracking.

With `OPENAI_API_KEY` and `WILDEDGE_DSN` configured, run from the SDK root:

```bash
uv run examples/openai_embeddings_example.py
```

This makes two small billable calls, one sync and one async. Find the printed
run ID in WildEdge and verify two inference events, each linked to its named
span, with batch size 2, dimensions, and input usage. Set
`OPENAI_EMBEDDING_MODEL` to override `text-embedding-3-small`.
