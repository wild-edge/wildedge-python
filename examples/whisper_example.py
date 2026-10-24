# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""Transcribe a WAV using a local Whisper model with automatic instrumentation.

Set WILDEDGE_DSN, then add the backend you want to run:
    uv run --with mlx-whisper examples/whisper_example.py mlx_whisper MODEL AUDIO
    uv run --with faster-whisper examples/whisper_example.py faster_whisper MODEL AUDIO

MODEL is a local path or Hugging Face repo, AUDIO is a WAV file.

See docs/whisper.md for lifecycle and explicit attachment examples.
"""

import argparse

import wildedge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["mlx_whisper", "faster_whisper"])
    parser.add_argument("model")
    parser.add_argument("audio")
    args = parser.parse_args()
    client = wildedge.init(integrations=[args.backend])
    try:
        with client.trace(), client.span(kind="agent_step", name="transcribe"):
            if args.backend == "mlx_whisper":
                import mlx_whisper  # noqa: PLC0415 -- optional runtime

                result = mlx_whisper.transcribe(args.audio, path_or_hf_repo=args.model)
                print(result["text"])
            else:
                from faster_whisper import (  # noqa: PLC0415 -- optional runtime
                    WhisperModel,
                )

                model = WhisperModel(args.model, device="cpu", compute_type="int8")
                segments, _info = model.transcribe(args.audio)
                print("".join(segment.text for segment in segments))
        client.flush()
    finally:
        client.close()


if __name__ == "__main__":
    main()
