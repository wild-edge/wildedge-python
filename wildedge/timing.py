"""Timing utilities."""

from __future__ import annotations

import time


def elapsed_ms(t0: float) -> int:
    """Return milliseconds elapsed since t0 (from time.perf_counter())."""
    return round((time.perf_counter() - t0) * 1000)


class Timer:
    """Context manager that measures elapsed wall-clock time in milliseconds.

    Usage::

        with Timer() as t:
            do_work()
        print(t.elapsed_ms)  # int, always set after the block exits
    """

    __slots__ = ("_t0", "elapsed_ms")

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = elapsed_ms(self._t0)
