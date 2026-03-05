from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any

from wildedge import config
from wildedge.timing import elapsed_ms

if TYPE_CHECKING:
    from wildedge.model import ModelHandle


class track:
    """
    Decorator and context manager for tracking model inference.

    As a decorator::

        @wildedge.track(handle, input_type="text", output_type="text")
        def generate(prompt):
            return model(prompt)

    As a context manager::

        with wildedge.track(handle) as t:
            result = model(input)

    """

    def __init__(
        self,
        handle: ModelHandle,
        *,
        input_type: str = "structured",
        output_type: str = "structured",
        capture_errors: bool = True,
        input_meta: Any = None,
        output_meta: Any = None,
        generation_config: Any = None,
    ):
        self.handle = handle
        self.input_type = input_type
        self.output_type = output_type
        self.capture_errors = capture_errors
        self.input_meta = input_meta
        self.output_meta = output_meta
        self.generation_config = generation_config
        self.start_time: float | None = None

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                self.handle.track_inference(
                    duration_ms=elapsed_ms(t0),
                    input_modality=self.input_type,
                    output_modality=self.output_type,
                    success=True,
                    input_meta=self.input_meta,
                    output_meta=self.output_meta,
                    generation_config=self.generation_config,
                )
                return result
            except Exception as exc:
                if self.capture_errors:
                    self.handle.track_error(
                        error_code="UNKNOWN",
                        error_message=str(exc)[: config.ERROR_MSG_MAX_LEN],
                    )
                raise

        return wrapper

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is None:
            return False
        duration_ms = elapsed_ms(self.start_time)
        if exc_type is not None:
            if self.capture_errors:
                self.handle.track_error(
                    error_code="UNKNOWN",
                    error_message=str(exc_val)[: config.ERROR_MSG_MAX_LEN]
                    if exc_val
                    else None,
                )
        else:
            self.handle.track_inference(
                duration_ms=duration_ms,
                input_modality=self.input_type,
                output_modality=self.output_type,
                success=True,
                input_meta=self.input_meta,
                output_meta=self.output_meta,
                generation_config=self.generation_config,
            )
        return False
