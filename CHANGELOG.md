# Changelog

Notable changes to wildedge-sdk. Every behavior change lands here; entries go
under Unreleased and move into a version section at release time.

## Unreleased

## 0.3.0 - 2026-10-21

### Added

- `mlx_whisper` and `faster_whisper` integrations track local Whisper model
  loads and transcriptions, including faster-whisper's
  `BatchedInferencePipeline`. Transcriptions record duration, language, word
  count and audio metadata. Audio and transcript attachments must be supplied
  explicitly through the manual tracking API. See docs/whisper.md.
- The OpenAI integration now tracks `responses.create()` for synchronous and
  asynchronous clients, including streaming. Records input/output token usage,
  cached and reasoning tokens, response status or incomplete reason, resolved
  model metadata, and time to first output delta for streams. Older OpenAI SDKs
  without Responses remain supported.
- OpenAI sync and async `embeddings.create()` instrumentation records latency,
  input token usage, batch size, vector dimensions, model metadata and errors.
  Embedding vectors and raw inputs are never attached by this adapter.
  `EmbeddingOutputMeta` now accepts optional `tokens_in` for analytics.
- Anthropic native `messages.stream()` helpers are now instrumented for sync
  and async clients, including event iteration, `text_stream`, and final-message
  helpers, with the shared streaming lifecycle contract.
- Process-wide integrations and hub patches now support sequential SDK client
  replacement after the previous owner closes. Reinitialization replaces a
  closed default. Existing models and provider clients keep their original
  destination and must be recreated. Attempts to use a closed telemetry
  destination warn once instead of silently dropping inference and error
  events.

### Changed

- The llama.cpp Python integration is now named `llama_cpp` in Python, CLI,
  and environment configuration. `gguf` remains a compatibility alias; `all`
  selects the integration once under its new name. Model format metadata
  remains `gguf`.
- Automatic integrations no longer attach raw payloads, including GGUF prompts,
  chat messages and replies. `attachments_enabled=True` enables uploads only for
  attachments explicitly passed to `track_inference(attachments=[...])`.
  Apps relying on automatic GGUF capture must supply attachments explicitly.
- `wildedge.init(dsn=...)` reuses the existing default client when the DSN
  points to the same project, for example `dsn=os.environ["WILDEDGE_DSN"]`
  under `wildedge run`. It used to create a second client.
- Enabling an integration or hub on a second client while another live client
  owns process-wide instrumentation now logs a warning. Automatic events keep
  going to the owner's project, as before, but this used to happen without a
  word. With `wildedge run --strict-integrations` it raises `RuntimeError`.
- Quantization labels are defined once in `wildedge.quantizations` and used
  by every integration. Behaviour changes:
  - MLX-LM reports N-bit weights as `int4`/`int8`, like Transformers, instead
    of `q4`/`q8`.
  - GGUF models labelled from `general.file_type` report the type name (for
    example `q4_k_m`) instead of its number, and the `i8` filename suffix
    reports `int8`.
  - faster-whisper reports weight precision (`int8`, `f16`) instead of
    CTranslate2 compute types such as `int8_float32`.
- The ultralytics integration now requires ultralytics 8.0.200 or later
  (was 8.0.0). Earlier 8.0.x releases keep their helpers under
  `ultralytics.yolo.utils`, and the integration turned itself off on them
  without a word. `instrument("ultralytics")` now logs a warning naming the
  installed version instead.

### Fixed

- The llama.cpp integration now tracks `Llama.create_chat_completion()` and
  `Llama.create_completion()`, streaming or not. It used to patch only
  `Llama.__call__`, so chat apps on llama-cpp-python reported model loads but
  no inference events. It now patches `create_completion`, which every entry
  point runs, so each call is tracked once.
- llama.cpp inference events now include the stop reason for non-streamed
  calls, the generation config (`temperature`, `top_p`, `top_k`, `max_tokens`,
  `repeat_penalty`, penalties, `seed`, stop sequence count), and, for chat
  calls, `prompt_type="chat"` and `turn_index` measured on the last user
  message.
- `import wildedge` no longer imports every installed ML framework and API
  client. Each library is imported when its integration is activated or a
  model from it is registered, so SDK import is faster and a broken unrelated
  package no longer stops the SDK from starting.
- Changing a model's metadata (for example `register_model(..., version="v2")`)
  now takes effect after a restart. With offline persistence on (the default),
  the model registry saved by an earlier process kept the first metadata ever
  registered for each model on that machine, so later changes were ignored.
  This includes anthropic models already saved as `"unknown"`.
- MLX-LM models now report the device they run on (`mps` for the Apple GPU)
  instead of always `cpu`. Accelerator names are defined once in
  `wildedge.accelerators`, and PyTorch, Transformers, Ultralytics,
  TensorFlow and Keras share their device detection.
- timm models now use the requested architecture/variant name for model identity
  instead of the Python class name, so models such as resnet18 and resnet50 no
  longer share telemetry. Manual registration falls back to timm architecture
  metadata, and explicit model IDs retain priority.
- Ultralytics `model.predict()` now records inference telemetry. Instrumentation
  patches `predict`, which `model(...)` delegates to, so both entry points are
  tracked exactly once.
- Streaming wrappers now finalize once across exhaustion, explicit closure,
  cancellation, and iterator errors. Early closure/cancellation records an
  unsuccessful inference with partial metadata when available. TTFT ignores
  lifecycle/role-only chunks; telemetry callbacks cannot break iteration and
  retain the trace context captured at stream creation. Remote API streams
  ending without their completion marker record `STREAM_INCOMPLETE`.
- A span attribute whose value is a list, dict or other non-scalar no longer
  gets the whole event batch rejected. Ingest only accepts scalar attribute
  values and answered 422 for the batch, dropping unrelated loads, inferences
  and spans. Such values are now sent as JSON strings, with one warning per
  key.
- `close(timeout=...)` and `flush(timeout=...)` now upload buffered
  attachments in the time left after sending events. `close()` used to stop
  the attachment uploader without a final upload, so a short-lived process
  lost the attachments of its last few seconds (the uploader runs every 5 s)
  while its events still referenced them. `close()` without a timeout still
  leaves them on disk for the next start.
- ONNX models exported with the default `main_graph` graph name (newer
  `torch.onnx.export`, Piper TTS voices) are now named after their file, so
  different voices no longer show up as one model called `main_graph`. A
  session created without an `id` falls back to the file name too, and
  `model_version` 0, negative or INT64_MAX (the exporters' "unset") is sent as
  `unknown` instead of `9223372036854775807`.
- transformers pipelines called with a string or a list record one inference
  event per call instead of two. `pipeline()` loads the model before
  `Pipeline.__init__` runs, so the model was registered with its own forward
  hooks, and each pipeline call was tracked by the pipeline and again by the
  model inside it. That model's hooks now skip forward passes inside a
  tracked call of its pipeline; other models the pipeline runs, and direct
  calls to the model, are still tracked. Generator and dataset inputs, which
  run lazily, are not covered yet (#78).
- `flush()` no longer sends a batch the background sender is already sending.
  Both used to send the same events and then each removed a batch from the
  queue, so events were duplicated and, depending on timing, later events
  could be removed without being sent. This showed up when a process started
  with events saved by an earlier run and flushed soon after.
- Enabling the `huggingface` hub no longer imports unused transformers model
  modules. To find references to `hf_hub_download` and `snapshot_download`,
  the hub tracker looked up the name on every loaded module, and transformers'
  lazy modules import code on lookup. With transformers loaded this imported
  about 140 extra modules, took about 0.15 s and logged about 430 "failed to
  patch" debug lines. The tracker now reads each module's namespace instead.

## 0.2.2 - 2026-09-30

### Added

- Error events carry `http_status` and `provider_error_code` for failed remote
  model API calls, and `track_error()` accepts both. The provider code comes
  from the provider's error body: OpenAI's code or type, Anthropic's error type
  (`overloaded_error`), or OpenRouter's numeric code. Wild Edge derives the
  error code (`AUTH_FAILED`, `QUOTA_EXCEEDED`, `RATE_LIMITED`, `BAD_REQUEST`,
  `PROVIDER_ERROR`) from them; this needs a server with that support.
- Error events from failed API calls carry `duration_ms`, the time until the
  call failed, so a rejected key and a 30 s timeout can be told apart;
  `track_error()` accepts it.
- `ErrorCode.CONNECTION_ERROR`.

### Changed

- The openai and anthropic integrations no longer record every failed call as
  `UNKNOWN`. When the provider responded, the event is `UNKNOWN` with
  `http_status` and `provider_error_code` for the server to classify, including
  errors raised mid-stream. Timeouts are `INFERENCE_TIMEOUT` and connection
  failures `CONNECTION_ERROR`.
- `wildedge.llm_api()` follows the same rules. It previously used the exception
  class name as the error code for every failure (for example
  `HTTPStatusError` or `ReadTimeout`); the class name is now used only for
  failures that are neither a provider response, a timeout nor a connection
  failure.

### Fixed

- Models tracked by the anthropic integration register with `model_format`
  `"api"`, like the openai integration. The Anthropic extractor was never in
  the client's extractor list, so they were registered as `"unknown"`.

## 0.2.1 - 2026-09-01

### Added

- `docs/providers.md`: the full table of recognized LLM API providers, the
  `model_source` each base URL resolves to, and how unrecognized/self-hosted
  hosts are named. Shipped in the release `llms.txt` / `llms-full.txt` assets.
- A test that keeps `docs/providers.md` in sync with the provider mapping in
  `wildedge.integrations.common`: every documented base URL must resolve to the
  documented source, and every known source must be documented.

### Changed

- README quickstart shows a mixed local + remote pipeline and points at the new
  provider docs; OpenAI-compatible hosts are documented as working through the
  `openai` integration via `base_url`. Dropped the stale pre-release notice.

## 0.2.0 - 2026-07-14

### Added

- `register_model()` accepts a `model_format` override for models without a matching extractor; `llm_api()` registers its models as format `"api"`, matching the openai/anthropic integrations instead of `"unknown"`.
- `source_from_base_url()` recognizes more provider hosts (anthropic, mistral, groq, together, deepseek, xai, google, fireworks, cerebras, perplexity, nvidia, huggingface, baseten) instead of falling back to raw hostnames, including suffix matching for per-resource subdomains (Azure OpenAI, Baseten).
- `examples/llm_api_example.py`: raw-HTTP LLM tracking with `wildedge.llm_api()`; existing examples updated to the module-level API (`wildedge.span` / `wildedge.flush` / `wildedge.register_model` instead of threading a client variable)
- Releases ship `llms.txt` and `llms-full.txt` as GitHub release assets: the full documentation for that exact version in one file, generated by `scripts/build_llms_txt.py`. README quickstart rewritten around the module-level API.
- `wildedge doctor --send-test-event`: sends one real span event through the full pipeline and reports the ingest response, proving DSN auth and connectivity end to end. The report gains an `environment` section (`WILDEDGE_*` variables, autoload PYTHONPATH status) plus `config_status` / `connectivity_status` fields.
- `wildedge.llm_api()`: provider-agnostic tracking for LLM calls made with any HTTP client (OpenRouter, vLLM, Ollama, OpenAI/Anthropic-compatible endpoints). Times the block, normalizes usage payloads from either provider shape via `call.usage()` / `call.response()`, supports TTFT marks and async use, and records exceptions as error events. See `docs/llm_api.md`.
- `register_model()` without a matching extractor defaults the model name to the explicit `model_id` instead of the placeholder object's type name.
- Process-wide default client: `wildedge.get_client()`, `wildedge.set_default_client()`, and module-level `trace`, `span`, `track_span`, `register_model`, `flush` delegating to it (#41)
- `init()` reuses the default client installed by `wildedge run` unless `dsn` is passed, so CLI and in-process init share one client (#41)
- `SpanContextManager.set_attributes()` and `fail()` for recording outcomes on an open span (#41)
- `wildedge run --strict` (env `WILDEDGE_STRICT`): exit with a reserved code instead of running untracked when bootstrap fails (120 config error, 122 internal error)
- Lazy default-client creation honors `WILDEDGE_INTEGRATIONS` and `WILDEDGE_HUBS` from the environment (#41)
- `docs/deployment.md`: the deployment contract (no-DSN behavior, strict mode, fork/exec servers, one client per process)

### Changed

- `wildedge doctor` exit codes are now differentiated: 0 pass, 1 configuration or dependency failure, 2 connectivity failure (`--network-check` or `--send-test-event`). A failing network check previously exited 1.
- The test suite isolates all default SDK state paths under tmp; tests no longer write to the machine-global state directory.

- `--strict-integrations` now takes effect: a failed required integration exits the process with code 121. Previously the enforcing code path was never invoked, so the flag was silently ignored.
- `--print-startup-report` and `--no-propagate` now work under `wildedge run`; both were only wired to the unused runner path before.
- Constructing a client without a DSN logs once per process at INFO. Previously every construction logged a WARNING.

### Removed

- `wildedge.runtime.runner`: an alternative bootstrap entry point that `wildedge run` never invoked. `sitecustomize` is the single bootstrap path; the runner's exit codes moved to `wildedge.runtime.bootstrap` and now apply under `--strict`.

## 0.1.5 - 2026-06-23

- Opt-in inference attachments (raw input/output upload)
- Fixed accelerator detection wiring; macOS CPU frequency and thermal sampling

## 0.1.4 and earlier

Predate this changelog; see the git history.
