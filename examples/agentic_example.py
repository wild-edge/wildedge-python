# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "openai"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""Agentic workflow example with tool use.

Demonstrates WildEdge tracing for a simple agent that:
  - Runs within a trace (one per agent session)
  - Wraps each reasoning step in an agent_step span
  - Wraps each tool call in a tool span
  - Tracks LLM inference automatically via the OpenAI integration

Run with: uv run agentic_example.py
Requires: OPENROUTER_API_KEY environment variable. Set WILDEDGE_DSN to send events.
"""

import json
import os
import time
import uuid

from openai import OpenAI

import wildedge

we = wildedge.init(
    app_version="1.0.0",
    integrations="openai",
)

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# --- Tools -------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    },
]


def get_weather(city: str) -> str:
    # ~150ms to simulate a real weather API call.
    time.sleep(0.15)
    return json.dumps({"city": city, "temperature_c": 18, "condition": "partly cloudy"})


def calculator(expression: str) -> str:
    # ~60ms to simulate a remote computation call.
    time.sleep(0.06)
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_HANDLERS = {
    "get_weather": get_weather,
    "calculator": calculator,
}


# --- Agent loop --------------------------------------------------------------


def call_tool(name: str, arguments: dict) -> str:
    with we.span(
        kind="tool",
        name=name,
        input_summary=json.dumps(arguments)[:200],
    ) as span:
        result = TOOL_HANDLERS[name](**arguments)
        span.output_summary = result[:200]
    return result


def retrieve_context(query: str) -> str:
    """Fetch relevant context from the vector store (~120ms)."""
    with we.span(
        kind="retrieval",
        name="vector_search",
        input_summary=query[:200],
    ) as span:
        time.sleep(0.12)
        result = f"[context: background knowledge relevant to '{query[:40]}']"
        span.output_summary = result
    return result


def run_agent(task: str, step_index: int, messages: list) -> str:
    # Fetch context before the first reasoning step, include it in the user turn.
    context = retrieve_context(task)
    messages.append({"role": "user", "content": f"{task}\n\nContext: {context}"})

    while True:
        with we.span(
            kind="agent_step",
            name="reason",
            step_index=step_index,
            input_summary=task[:200],
        ) as span:
            response = openai_client.chat.completions.create(
                model="qwen/qwen3.5-flash-02-23",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=512,
            )
            choice = response.choices[0]
            span.output_summary = choice.finish_reason

        messages.append(choice.message.model_dump(exclude_none=True))

        if choice.finish_reason == "tool_calls":
            step_index += 1
            for tool_call in choice.message.tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                result = call_tool(tool_call.function.name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
                # Not instrumented: context window update between tool calls (~80ms).
                # Shows up as a gap stripe in the trace view.
                time.sleep(0.08)
        else:
            return choice.message.content or ""


# --- Main --------------------------------------------------------------------

TASKS = [
    "What's the weather like in Tokyo, and what is 42 * 18?",
    "Is it warmer in Paris or Berlin right now?",
]

system_prompt = "You are a helpful assistant. Use tools when needed."
messages = [{"role": "system", "content": system_prompt}]

with we.trace(agent_id="demo-agent", run_id=str(uuid.uuid4())):
    for i, task in enumerate(TASKS, start=1):
        print(f"\nTask {i}: {task}")
        reply = run_agent(task, step_index=i, messages=messages)
        print(f"Reply: {reply}")

we.flush()
