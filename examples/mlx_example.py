# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk", "mlx-lm"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""
MLX / mlx-lm integration example — Apple Silicon only.

WildEdge patches mlx_lm.load and mlx_lm.generate at client initialisation.
Load timing, HuggingFace Hub download tracking, inference metrics (tokens/sec,
token counts), and unload tracking all happen automatically.

Usage:
    uv run mlx_example.py
    uv run mlx_example.py --model mlx-community/Llama-3.2-1B-Instruct-4bit
    uv run mlx_example.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
"""

from __future__ import annotations

import argparse

import mlx_lm

import wildedge

PROMPTS = [
    "Explain on-device ML inference in one sentence.",
    "What makes Apple Silicon well-suited for local AI?",
    "Name three advantages of privacy-preserving inference.",
]

DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def main() -> None:
    parser = argparse.ArgumentParser(description="WildEdge + mlx-lm example")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace repo or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="Max tokens to generate per prompt (default: 80)",
    )
    args = parser.parse_args()

    # instrument() patches mlx_lm.load and mlx_lm.generate — must be called
    # before any model is loaded.
    client = wildedge.WildEdge(app_version="1.0.0")  # set WILDEDGE_DSN env var
    client.instrument("mlx", hubs=["huggingface"])

    print(f"\nLoading {args.model} ...")
    model, tokenizer = mlx_lm.load(args.model)  # load + download tracked automatically

    print(f"\nRunning {len(PROMPTS)} prompts (max_tokens={args.max_tokens}):\n")
    for i, prompt in enumerate(PROMPTS, 1):
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        )
        print(f"[{i}] Q: {prompt}")
        print(f"    A: {response}\n")

    client.flush()
    print("Done — events flushed to WildEdge.")


if __name__ == "__main__":
    main()
