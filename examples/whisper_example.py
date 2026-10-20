"""Transcribe a WAV using a local Whisper model with automatic instrumentation.

Install mlx-whisper (Apple Silicon) or faster-whisper, set WILDEDGE_DSN, then:
    python examples/whisper_example.py mlx_whisper /path/to/model /path/to/audio.wav
    python examples/whisper_example.py faster_whisper /path/to/model /path/to/audio.wav

See docs/whisper.md for lifecycle and opt-in attachment capture behavior.
"""

import argparse

import wildedge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["mlx_whisper", "faster_whisper"])
    parser.add_argument("model")
    parser.add_argument("audio")
    parser.add_argument(
        "--capture", action="store_true", help="Attach audio and transcript"
    )
    args = parser.parse_args()
    client = wildedge.init(
        integrations=[args.backend], attachments_enabled=args.capture
    )
    try:
        with client.trace(), client.span(name="transcribe"):
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
