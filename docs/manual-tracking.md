# Manual tracking

Use manual tracking when your framework is not covered by a WildEdge integration, or when you need richer metadata than auto-instrumentation provides.

## When to use it

- Your model class is custom (e.g. a `torch.nn.Module` subclass not loaded via `timm` or `transformers`)
- You are calling a remote API not yet covered by an integration
- You want to attach input/output metadata (token counts, image dimensions, confidence scores, etc.)
- You want to record user feedback tied to a specific inference

## torch and keras

For `torch` and `keras`, models are user-defined subclasses with no constructor to patch. Use `client.load()` to get load, unload, and inference tracking automatically:

```python
model = client.load(MyModel)
output = model(x)  # tracked automatically
```

## Register your model

Every model needs a handle before you can track events against it. Pass the model object and an explicit `model_id`:

```python
import wildedge

client = wildedge.init()  # uses WILDEDGE_DSN if set; otherwise no-op

# Optional: enable auto-instrumentation alongside manual tracking.
# client = wildedge.init(integrations=["transformers"], hubs=["huggingface"])

handle = client.register_model(
    my_model,
    model_id="my-org/my-model",
    source="local",        # where the weights came from
    family="resnet",       # optional model family
    version="1.0.0",       # optional version string
    quantization="int8",   # optional quantization label
)
```

For remote APIs with no local object to inspect, pass a placeholder:

```python
handle = client.register_model(
    object(),
    model_id="openai/gpt-4o",
    source="https://api.openai.com",
    family="gpt-4o",
    version="2024-08-06",
)
```

`register_model` is idempotent - calling it twice with the same `model_id` returns the existing handle.

## Track inference

### Decorator

WildEdge times the function and records success or error automatically:

```python
@wildedge.track(handle, input_type="text", output_type="text")
def generate(prompt: str) -> str:
    return my_model(prompt)
```

### Context manager

Use this when you need access to the result before emitting the event, or when the tracked block is not a standalone function:

```python
with wildedge.track(handle, input_type="image", output_type="structured"):
    result = my_model(image_tensor)
```

### Direct call

For full control over metadata - token counts, confidence scores, generation config, etc.:

```python
from wildedge import GenerationConfig, GenerationOutputMeta, TextInputMeta
from wildedge.timing import Timer

with Timer() as t:
    result = my_model(prompt)

inference_id = handle.track_inference(
    duration_ms=t.elapsed_ms,
    input_modality="text",
    output_modality="generation",
    success=True,
    input_meta=TextInputMeta(
        token_count=len(prompt.split()),
        prompt_type="chat",
    ),
    output_meta=GenerationOutputMeta(
        tokens_in=input_tokens,
        tokens_out=output_tokens,
        tokens_per_second=round(output_tokens / t.elapsed_ms * 1000, 1),
        stop_reason="stop",
    ),
    generation_config=GenerationConfig(
        temperature=0.7,
        max_tokens=512,
    ),
)
```

`track_inference` returns an `inference_id` string you can use to attach feedback later.

## Input and output metadata

All fields are optional.

### Text input (`TextInputMeta`)

| Field | Type | Description |
|---|---|---|
| `char_count` | `int` | Character count |
| `word_count` | `int` | Word count |
| `token_count` | `int` | Token count |
| `language` | `str` | BCP-47 language code |
| `prompt_type` | `str` | e.g. `"chat"`, `"completion"`, `"instruct"` |
| `turn_index` | `int` | Position in a multi-turn conversation |
| `contains_code` | `bool` | Whether the input contains code |

### Image input (`ImageInputMeta`)

| Field | Type | Description |
|---|---|---|
| `width` | `int` | Pixel width |
| `height` | `int` | Pixel height |
| `channels` | `int` | Channel count |
| `format` | `str` | e.g. `"jpeg"`, `"png"`, `"rgb"` |
| `source` | `str` | e.g. `"camera"`, `"file"`, `"stream"` |

### Audio input (`AudioInputMeta`)

| Field | Type | Description |
|---|---|---|
| `duration_ms` | `int` | Audio length |
| `sample_rate` | `int` | Hz |
| `channels` | `int` | Channel count |
| `format` | `str` | e.g. `"wav"`, `"mp3"` |
| `is_streaming` | `bool` | Live stream vs pre-recorded |

### Generation output (`GenerationOutputMeta`)

| Field | Type | Description |
|---|---|---|
| `tokens_in` | `int` | Prompt token count |
| `tokens_out` | `int` | Completion token count |
| `tokens_per_second` | `float` | Generation throughput |
| `cached_input_tokens` | `int` | Tokens served from KV cache |
| `reasoning_tokens_out` | `int` | Reasoning/thinking tokens (e.g. o1) |
| `time_to_first_token_ms` | `int` | TTFT latency |
| `stop_reason` | `str` | e.g. `"stop"`, `"length"`, `"content_filter"` |

### Classification output (`ClassificationOutputMeta`)

| Field | Type | Description |
|---|---|---|
| `num_predictions` | `int` | Number of predictions returned |
| `avg_confidence` | `float` | Mean confidence across predictions |
| `top_k` | `list[TopKPrediction]` | Top-k labels with confidence scores |

### Detection output (`DetectionOutputMeta`)

| Field | Type | Description |
|---|---|---|
| `num_predictions` | `int` | Number of detections |
| `avg_confidence` | `float` | Mean confidence |
| `top_k` | `list[TopKPrediction]` | Top detections with bounding boxes |

### Embedding output (`EmbeddingOutputMeta`)

| Field | Type | Description |
|---|---|---|
| `dimensions` | `int` | Embedding vector size |

## Track errors

```python
try:
    result = my_model(input)
except Exception as exc:
    handle.track_error(
        error_code="UNKNOWN",
        error_message=str(exc),
    )
    raise
```

`wildedge.track` (decorator and context manager) captures errors automatically when `capture_errors=True` (the default).

## Track feedback

Link user or automated feedback to a specific inference via the `inference_id` returned by `track_inference`:

```python
from wildedge import FeedbackType

inference_id = handle.track_inference(duration_ms=..., ...)

# Later, when feedback is available
handle.track_feedback(inference_id, FeedbackType.THUMBS_UP)
```

If you always want to attach feedback to the most recent inference on a handle, use the shorthand:

```python
handle.feedback(FeedbackType.THUMBS_DOWN)
```

`FeedbackType` values: `THUMBS_UP`, `THUMBS_DOWN`.

## Track spans for agentic workflows

Use span events to track non-inference steps like planning, tool calls, retrieval, or memory updates.

```python
from wildedge.timing import Timer

with Timer() as t:
    tool_result = call_tool()

client.track_span(
    kind="tool",
    name="call_tool",
    duration_ms=t.elapsed_ms,
    status="ok",
    attributes={"tool": "search"},
)
```

You can also attach optional correlation fields (`trace_id`, `span_id`,
`parent_span_id`, `run_id`, `agent_id`, `step_index`, `conversation_id`) to any
event by passing them into `track_inference`, `track_error`, `track_feedback`,
or `track_span`.

### Trace context helpers

Use `trace_context()` and `span_context()` to auto-populate correlation fields
for all events emitted inside the block:

```python
import wildedge

with wildedge.trace_context(run_id="run-123", agent_id="agent-1"):
    with wildedge.span_context(step_index=1):
        handle.track_inference(duration_ms=12)
```
