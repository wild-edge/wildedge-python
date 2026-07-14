# Tracking LLM calls without a client library

`wildedge.llm_api()` records LLM API calls made with any HTTP client: httpx or
requests against OpenRouter, vLLM, Ollama, or any OpenAI-compatible or
Anthropic-compatible endpoint. Use it when auto-instrumentation does not
apply because your app does not use the `openai` or `anthropic` packages. If
it does use them, prefer the integrations; they capture the same events with
zero code.

The name is deliberate: this tracks calls to an LLM behind an *API boundary*.
An LLM running inside your own process (llama.cpp, transformers, MLX) is
covered by the [framework integrations](manual-tracking.md), which also
capture what no API boundary can expose: quantization from the model
artifact, memory footprint, load/unload timing, and hardware linkage.

The block is timed automatically, events correlate with surrounding
`wildedge.trace()` / `wildedge.span()` blocks, and everything is a silent
no-op without a DSN.

## Quickstart

```python
import httpx
import wildedge

async def generate(prompt: str, model: str) -> dict:
    with wildedge.llm_api(model=model, provider="openrouter", prompt=prompt) as call:
        async with httpx.AsyncClient(timeout=300) as http:
            response = await http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            )
        data = response.json()
        call.response(data)
    return data
```

`call.response()` pulls token usage, stop reason, and API metadata from the
full response payload, dict or SDK object, OpenAI shape
(`usage.prompt_tokens`, `choices[0].finish_reason`) or Anthropic shape
(`usage.input_tokens`, top-level `stop_reason`).

Runnable version: [examples/llm_api_example.py](../examples/llm_api_example.py),
stdlib urllib against OpenRouter, no client library at all.

## Recording pieces individually

When you do not have a full response payload, set what you know:

```python
with wildedge.llm_api(model="gemma-7b", base_url="http://localhost:11434") as call:
    result = post_to_ollama(...)
    call.usage(result["usage"])            # payload in either provider shape
    call.usage(tokens_in=52, tokens_out=209)  # or explicit fields; these win
    call.stop_reason = "stop"
    call.first_token()                     # TTFT mark for streaming
    call.success = False                   # delivered but unusable output
```

An exception escaping the block records an error event with the exception
class as the error code, and no inference event.

## Model identity

Events register under `model` as the model id. `provider` names the source
directly; alternatively `base_url` derives it (`openrouter.ai` becomes
`openrouter`, unknown hosts record the hostname). Pass `prompt` or `messages`
(chat format) to attach input metadata such as prompt length.

## Agentic pipelines

Combine with traces for multi-step visibility:

```python
with wildedge.trace(run_id=run_id, agent_id="skin-generator"):
    for attempt in range(2):
        with wildedge.span(kind="agent_step", name="generate", step_index=attempt):
            with wildedge.llm_api(model=model, provider="openrouter", prompt=prompt) as call:
                data = await post_chat_completion(prompt)
                call.response(data)
```
