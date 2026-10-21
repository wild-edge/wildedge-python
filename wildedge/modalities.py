"""Attachment modalities: the kind of data an attachment holds.

``attachment_modalities`` lists the ones a client may upload, as ``Modality``
members or their string values (``"audio"``), which is what the env var and
CLI use. Unlike the accelerator and quantization labels, this set is closed.
The client rejects names it doesn't know instead of silently uploading nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    OTHER = "other"  # any content type not covered above, such as tensors


ALL: tuple[str, ...] = tuple(modality.value for modality in Modality)

# Without an explicit choice only text is uploaded. Voice and images can
# identify a person, so they are sent only when the app opts in.
DEFAULT: tuple[str, ...] = (Modality.TEXT.value,)

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
MEDIA_TOP_LEVEL_TYPES = frozenset(
    {
        Modality.TEXT.value,
        Modality.IMAGE.value,
        Modality.AUDIO.value,
        Modality.VIDEO.value,
    }
)


def normalize(modality: Modality | str) -> str:
    """The plain string value, so enum members and strings compare alike."""
    try:
        return Modality(modality.strip().lower()).value
    except ValueError:
        raise ValueError(
            f"unknown attachment modality {modality!r}, "
            f"expected any of {', '.join(ALL)}"
        ) from None


def from_content_type(content_type: str) -> str:
    """The modality a MIME type belongs to, such as ``audio`` for ``audio/wav``."""
    media_type = content_type.split(";", 1)[0].strip().lower()
    top_level = media_type.partition("/")[0]
    if top_level in MEDIA_TOP_LEVEL_TYPES:
        return top_level
    if media_type in TEXT_APPLICATION_TYPES or media_type.endswith(TEXT_SUFFIXES):
        return Modality.TEXT.value
    return Modality.OTHER.value


def parse(value: str | Iterable[Modality | str]) -> frozenset[str]:
    """Validate a modality list. A string is comma-separated, and ``"all"``
    allows every modality."""
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return frozenset(ALL)
        value = [item for item in value.split(",") if item.strip()]
    return frozenset(normalize(item) for item in value)
