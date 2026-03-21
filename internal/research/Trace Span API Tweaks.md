# API Tweak Proposal: Trace and Span UX

## Background
The current API mixes module-level context managers (trace_context, span_context) with a client method for span emission (WildEdge.span). This works, but it is slightly asymmetric and makes usage feel less Pythonic. There is also easy confusion between correlation attributes (event-level attributes) and span attributes (span.attributes), because the API currently exposes both "attributes" and "span_attributes".

## Goals
- Make the API feel symmetric and Pythonic for common workflows.
- Reduce cognitive overhead for users who only need basic tracing.
- Keep event schema and protocol unchanged.
- Preserve backward compatibility with minimal code changes.

## Proposal
### 1) Add client-level trace context
Add a client method that mirrors WildEdge.span:

- WildEdge.trace(...) -> context manager that sets trace context
- Optional return value: TraceContext (for advanced use)

This makes usage consistent:

with we.trace(agent_id=..., conversation_id=...):
    with we.span(...):
        ...

### 2) Clarify attributes vs correlation
Rename or alias parameters to make intent explicit:

- WildEdge.span(..., attributes=...) refers to span attributes (stored under span.attributes)
- WildEdge.span(..., context=...) refers to correlation attributes (stored at the event root)

Backwards compatible behavior:
- span_attributes remains accepted as an alias for attributes
- attributes on trace_context remains accepted as an alias for context

### 3) Optional client-level span_context
Expose WildEdge.span_context(...) as a thin wrapper over span_context for symmetry. This is a low-level API for correlation without emitting a span event.

### 4) Optional decorator sugar (static cases)
Add @we.trace as an optional decorator for static metadata. This is explicit sugar only, not the recommended default for dynamic metadata:

@we.trace(agent_id="agent-1")
def handle_message(...):
    ...

## Example (proposed usage)

we = wildedge.WildEdge()
we.instrument("openai")

with we.trace(agent_id="agent-a", conversation_id=channel):
    with we.span(
        kind="agent_step",
        name="respond",
        step_index=step_index,
        attributes={"model": model},
    ):
        response = client.chat.completions.create(...)

## Migration and compatibility
- No protocol changes.
- Existing code using trace_context and span_context continues to work.
- Existing code using span_attributes continues to work.
- Add warnings only if we want to encourage migration; otherwise keep silent aliases.

## Notes for agntr
Once WildEdge.trace exists, agntr can switch from wildedge.trace_context(...) to we.trace(...) for symmetry, and use attributes= for span attributes.
