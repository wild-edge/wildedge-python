# Supporting Agentic Workflows

## Summary
WildEdge already captures model load/download/inference events and basic errors. Agentic systems introduce multi-step plans, tool calls, retrieval, memory, and feedback loops that require traceability across steps and richer metadata. This document proposes a full, production-grade extension to the SDK and protocol to support agentic workflows while remaining backward compatible.

## Goals
- Support end-to-end tracing of agent runs across LLM calls, tools, retrieval, and memory.
- Provide a minimal, ergonomic SDK API for trace/span creation and propagation.
- Preserve existing integrations and CLI behavior with no required app changes.
- Allow vendor-agnostic metadata attachment without frequent schema changes.
- Enable safe payload controls for privacy and compliance.

## Non-Goals
- Build a full UI or backend visualization in this repository.
- Require user code changes for existing integrations.
- Enforce a single agent framework or orchestration style.

## Current SDK Snapshot
- Events: model_load, model_unload, model_download, inference, error, feedback.
- Protocol envelope: batch with protocol_version, device, models, session_id, events.
- Integrations: onnx, gguf, openai, timm, tensorflow, transformers, ultralytics, mlx, torch/keras noop.
- Manual tracking: register_model + track_inference, track_error, track_feedback.
- CLI: wildedge run autoloads runtime and patches integrations.

## Requirements For Agentic Monitoring
- Correlate events into a single run with a DAG of spans.
- Capture agent steps and tool calls that are not model inference.
- Associate retrieval results, memory writes, and guardrails with the same run.
- Support async workloads and nested spans safely.
- Keep payload sizes and sensitive data under control.

## Proposed Protocol Extensions
### 1) Correlation Fields (Optional on All Events)
- trace_id: string
- span_id: string
- parent_span_id: string
- run_id: string
- agent_id: string
- step_index: integer
- conversation_id: string

### 2) Generic Attributes Map (Optional on All Events)
- attributes: object (string keys, JSON-serializable values)

### 3) New Span Event Type
- event_type: "span"
- span.kind: "agent_step" | "tool" | "retrieval" | "memory" | "router" | "guardrail" | "cache" | "eval" | "custom"
- span.name: string
- span.duration_ms: integer
- span.status: "ok" | "error"
- span.input_summary: string (optional, truncated)
- span.output_summary: string (optional, truncated)
- span.attributes: object

### 4) Optional Usage Block For Inference Events
- inference.usage.tokens_in: integer
- inference.usage.tokens_out: integer
- inference.usage.cached_input_tokens: integer
- inference.usage.reasoning_tokens_out: integer
- inference.usage.cost_usd: number

### 5) Protocol Versioning
- Bump PROTOCOL_VERSION to 1.1
- All new fields remain optional and are ignored by older backends.

## SDK API Additions
### Trace Context
- wildedge.trace(name: str, run_id: str | None, agent_id: str | None, conversation_id: str | None)
- wildedge.span(kind: str, name: str, attributes: dict | None)
- Context propagation via contextvars for async support.

### Span Emission
- Span context managers emit a SpanEvent on exit with duration and status.
- Current trace/span context attaches to all events automatically.

### Example
```python
import wildedge

with wildedge.trace("puzzle_run", run_id="run-123", agent_id="puzzle-agent"):
    with wildedge.span("agent_step", name="analyze_board", attributes={"iteration": 4}):
        result = planner()
    with wildedge.span("tool", name="rotate_cell", attributes={"col": 2, "row": 1}):
        apply_move()
```

## Integration Strategy
### Phase 1: Core SDK
- Add trace/span context and SpanEvent.
- Attach correlation fields to all existing events.
- Add generic attributes map.

### Phase 2: Framework Integrations
- langchain / langgraph
- llamaindex
- autogen
- crewai
- dspy
- litellm (covers many providers)

### Integration Behavior
- Wrap framework callbacks to create spans for steps, tools, retrieval, memory.
- Reuse existing inference tracking when LLM calls happen under the hood.
- Map framework-specific fields into attributes without schema changes.

## CLI Impact
- No changes required to wildedge run.
- New SDK features activate automatically when integrations emit spans.

## Privacy, Security, and Payload Controls
- Add settings to disable input/output capture globally.
- Redaction hooks for PII or secrets.
- Truncation and hashing for large content fields.
- Clear documentation and defaults that favor safety.

## Compatibility and Migration
- Existing clients continue to work unchanged.
- Server must accept optional new fields.
- Unknown event types should be ignored safely by older servers.

## Testing Plan
- Unit tests for trace context propagation and span emission.
- Integration tests for nested spans with async workloads.
- Golden tests for event JSON serialization and protocol versioning.

## Risks
- High cardinality from unbounded attributes.
- Large payloads from agent inputs/outputs.
- Trace context leaks across tasks if not isolated.

## Milestones
### M1: Protocol and Core SDK
- Add correlation fields and attributes map.
- Add SpanEvent and tracing API.
- Update documentation and examples.

### M2: First Framework Integration
- Implement langchain or llamaindex integration.
- Validate span graph integrity.

### M3: Broader Ecosystem
- Add autogen, crewai, dspy, litellm.
- Add configurable redaction and capture policies.

## Open Questions
- Should SpanEvent share top-level shape with InferenceEvent or be its own payload?
- Should span.status allow more than ok/error?
- Should run_id be auto-generated when a trace starts without explicit id?
- Should attributes enforce a key whitelist or allow arbitrary JSON?

