# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "transformers", "torch"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""
HuggingFace Transformers integration example.

WildEdge patches transformers.pipeline (and AutoModel.from_pretrained) at
client initialisation, so load timing, download tracking, inference tracking,
and unload tracking all happen automatically.

Usage:
    uv run transformers_example.py                     # text classification
    uv run transformers_example.py --task generate     # text generation
    uv run transformers_example.py --task embed        # feature extraction
"""

from __future__ import annotations

import argparse

from transformers import pipeline

import wildedge


def run_classify() -> None:
    pipe = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    inputs = [
        "I absolutely loved this film — the performances were outstanding!",
        "The service was awful and the food arrived cold.",
        "An average experience, nothing special either way.",
    ]
    print("Sentiment classification:")
    for text in inputs:
        result = pipe(text)
        label = result[0]["label"]
        score = result[0]["score"]
        bar = "█" * int(score * 20)
        print(f"  {label:<9} {bar:<20} {score:.3f}  {text!r}")


def run_generate() -> None:
    pipe = pipeline("text-generation", model="gpt2", max_new_tokens=40)
    prompts = [
        "The future of on-device AI is",
        "Once upon a time, a small robot learned",
    ]
    print("Text generation (GPT-2):")
    for prompt in prompts:
        result = pipe(prompt, do_sample=False)
        print(f"  Prompt : {prompt!r}")
        print(f"  Output : {result[0]['generated_text']!r}\n")


def run_embed() -> None:
    pipe = pipeline("feature-extraction", model="bert-base-uncased")
    sentences = [
        "Machine learning is transforming every industry.",
        "On-device inference keeps your data private.",
        "WildEdge monitors ML performance in production.",
    ]
    print("Feature extraction (BERT):")
    for sent in sentences:
        result = pipe(sent)
        # result shape: [1, seq_len, hidden_size]; take CLS token embedding
        cls_embedding = result[0][0]
        dims = len(cls_embedding)
        norm = sum(v**2 for v in cls_embedding) ** 0.5
        print(f"  dims={dims}  L2={norm:.2f}  {sent!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WildEdge + HuggingFace Transformers example"
    )
    parser.add_argument(
        "--task",
        choices=["classify", "generate", "embed"],
        default="classify",
        help="Pipeline task to demonstrate (default: classify)",
    )
    args = parser.parse_args()

    # instrument() patches transformers.pipeline and AutoModel.from_pretrained
    # before any model is loaded; everything below is tracked automatically.
    client = wildedge.WildEdge(app_version="1.0.0")  # set WILDEDGE_DSN env var
    client.instrument("transformers", hubs=["huggingface"])

    print()
    {"classify": run_classify, "generate": run_generate, "embed": run_embed}[
        args.task
    ]()

    client.flush()
    print("\nDone. Events flushed to WildEdge.")


if __name__ == "__main__":
    main()
