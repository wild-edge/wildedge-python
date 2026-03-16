from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from wildedge.platforms.hardware import HardwareContext

if TYPE_CHECKING:
    from wildedge.platforms.base import Platform


class HardwareSampler:
    """Samples hardware context on a background thread at a fixed interval.

    The platform is injected so this module has no dependency on platforms/__init__.py.
    The first snapshot is taken synchronously in start() so capture_hardware() never
    returns an empty context immediately after initialisation.
    """

    def __init__(self, platform: Platform, interval_s: float = 30.0):
        self.platform = platform
        self.interval_s = interval_s
        self.current: HardwareContext = HardwareContext()
        self.done = threading.Event()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="wildedge-hw-sampler"
        )

    def start(self) -> None:
        self.current = self.platform.hardware_context()
        self.thread.start()

    def stop(self) -> None:
        self.done.set()

    def snapshot(self) -> HardwareContext:
        return self.current

    def _run(self) -> None:
        while not self.done.wait(self.interval_s):
            self.current = self.platform.hardware_context()
