# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "openai"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""ChatGPT (OpenAI API): fully manual integration.

Shows how to instrument a remote LLM with no local model file.
Tracks input/output token counts, generation config, latency, errors,
and user feedback without any auto-instrumentation hooks.

Run with: uv run chatgpt_example.py
Requires: WILDEDGE_DSN and OPENAI_API_KEY environment variables.
"""

from openai import OpenAI

import wildedge
from wildedge import FeedbackType, GenerationConfig, GenerationOutputMeta, TextInputMeta
from wildedge.timing import Timer

MODEL = "gpt-4o"
MODEL_VERSION = "2024-08-06"

client = wildedge.WildEdge(
    app_version="1.0.0",  # set WILDEDGE_DSN env var
)

# Remote models have no local object to inspect, so register with a
# placeholder and supply all metadata explicitly.
handle = client.register_model(
    object(),
    model_id=f"openai/{MODEL}",
    source="https://api.openai.com",
    family="gpt-4o",
    version=MODEL_VERSION,
)

openai_client = OpenAI()  # set OPENAI_API_KEY env var or pass api_key= explicitly

prompts = [
    "Explain transformer attention in one sentence.",
    "What is the capital of Japan?",
    "Write a haiku about edge AI.",
]

temperature = 0.7
max_tokens = 256

for turn_index, prompt in enumerate(prompts):
    messages = [{"role": "user", "content": prompt}]

    try:
        with Timer() as t:
            response = openai_client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        usage = response.usage
        choice = response.choices[0]
        completion = choice.message.content or ""
        tokens_per_second = (
            round(usage.completion_tokens / t.elapsed_ms * 1000, 1)
            if usage.completion_tokens and t.elapsed_ms > 0
            else None
        )

        inference_id = handle.track_inference(
            duration_ms=t.elapsed_ms,
            input_modality="text",
            output_modality="text",
            success=True,
            input_meta=TextInputMeta(
                char_count=len(prompt),
                word_count=len(prompt.split()),
                token_count=usage.prompt_tokens,
                prompt_type="chat",
                turn_index=turn_index,
                contains_code="```" in prompt,
            ),
            output_meta=GenerationOutputMeta(
                tokens_in=usage.prompt_tokens,
                tokens_out=usage.completion_tokens,
                tokens_per_second=tokens_per_second,
                stop_reason=choice.finish_reason,
                context_used=usage.total_tokens,
            ),
            generation_config=GenerationConfig(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )

        print(f"Q: {prompt}\nA: {completion}\n")

        # Simulate feedback: short completions get a thumbs down.
        feedback_type = (
            FeedbackType.THUMBS_UP if len(completion) > 40 else FeedbackType.THUMBS_DOWN
        )
        handle.track_feedback(inference_id, feedback_type)

    except Exception as exc:
        handle.track_error(error_code="UNKNOWN", error_message=str(exc)[:200])
        raise

client.close(timeout=5.0)
