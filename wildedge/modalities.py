"""Attachment modalities: the kind of data an attachment holds.

``attachment_modalities`` lists the ones a client may upload. Unlike the
accelerator and quantization labels, this set is closed, because the client
rejects names it doesn't know instead of silently uploading nothing.
"""

from __future__ import annotations

from collections.abc import Iterable

TEXT = "text"
IMAGE = "image"
AUDIO = "audio"
VIDEO = "video"
OTHER = "other"  # any content type not covered above, such as tensors

ALL: tuple[str, ...] = (TEXT, IMAGE, AUDIO, VIDEO, OTHER)

# Without an explicit choice only text is uploaded. Voice and images can
# identify a person, so they are sent only when the app opts in.
DEFAULT: tuple[str, ...] = (TEXT,)

# Structured text formats under application/.
TEXT_APPLICATION_TYPES = frozenset(
    {
        "application/json",
        "application/x-ndjson",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
    }
)
TEXT_SUFFIXES = ("+json", "+xml")
MEDIA_TOP_LEVEL_TYPES = frozenset({TEXT, IMAGE, AUDIO, VIDEO})


def from_content_type(content_type: str) -> str:
    """The modality a MIME type belongs to, such as ``audio`` for ``audio/wav``."""
    media_type = content_type.split(";", 1)[0].strip().lower()
    top_level = media_type.partition("/")[0]
    if top_level in MEDIA_TOP_LEVEL_TYPES:
        return top_level
    if media_type in TEXT_APPLICATION_TYPES or media_type.endswith(TEXT_SUFFIXES):
        return TEXT
    return OTHER


def parse(value: str | Iterable[str]) -> frozenset[str]:
    """Validate a modality list; a string is comma-separated, ``"all"`` allows all."""
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return frozenset(ALL)
        value = value.split(",")
    names = frozenset(item.strip().lower() for item in value if item.strip())
    unknown = names - set(ALL)
    if unknown:
        raise ValueError(
            f"unknown attachment modality {', '.join(sorted(unknown))}; "
            f"expected any of {', '.join(ALL)} or 'all'"
        )
    return names
