# Changelog

Notable changes to wildedge-sdk. Every behavior change lands here; entries go
under Unreleased and move into a version section at release time.

## Unreleased

### Added

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
