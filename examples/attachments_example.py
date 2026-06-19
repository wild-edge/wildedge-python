# /// script
# requires-python = ">=3.10"
# dependencies = ["wildedge-sdk"]
#
# [tool.uv.sources]
# wildedge-sdk = { path = "..", editable = true }
# ///
"""
Attachment upload example. Run with: uv run attachments_example.py

Opt-in raw input/output capture. When `attachments_enabled=True` (and the
project has the paid feature turned on), the SDK buffers the raw bytes locally,
writes a reference into the inference event, and uploads the bytes independently
via a presigned URL, so the batch flush never waits on the upload.

Attachments are off by default and must be explicitly enabled. Set WILDEDGE_DSN
to see real uploads; otherwise the client runs in no-op mode.
"""

import wildedge
from wildedge import Attachment


# Optional: redact / drop attachments before they are buffered.
def redact(attachments: list[Attachment]) -> list[Attachment]:
    return [a for a in attachments if a.content_type != "application/secret"]


client = wildedge.init(
    app_version="1.0.0",
    attachments_enabled=True,
    max_attachments_per_inference=5,
    max_attachment_size_bytes=5 * 1024 * 1024,
    attachment_storage_strategy="file",  # or "inline" for small blobs
    attachment_filter=redact,
)

handle = client.register_model(
    object(),
    model_id="doc-classifier-v1",
    source="local",
    family="custom",
)

# Pretend these came from a real inference call.
image_bytes = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
answer = "This document is an invoice."

inference_id = handle.track_inference(
    duration_ms=120,
    input_modality="image",
    output_modality="text",
    attachments=[
        Attachment(content_type="image/jpeg", role="input", data=image_bytes),
        Attachment(content_type="text/plain", role="output", data=answer.encode()),
    ],
)

print(f"tracked inference {inference_id[:8]}… with 2 attachments")

# Bytes upload in the background; flush/close lets buffered events drain.
client.close()
