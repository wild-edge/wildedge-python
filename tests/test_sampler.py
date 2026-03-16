from __future__ import annotations

from unittest.mock import MagicMock, patch

from wildedge.platforms import (
    capture_hardware,
    is_sampling,
    start_sampler,
    stop_sampler,
)
from wildedge.platforms.hardware import HardwareContext
from wildedge.platforms.sampler import HardwareSampler


def make_platform(snapshot: HardwareContext | None = None) -> MagicMock:
    platform = MagicMock()
    platform.hardware_context.return_value = snapshot or HardwareContext(
        memory_available_bytes=1_000_000
    )
    return platform


class TestHardwareSampler:
    def test_snapshot_is_warm_after_start(self):
        platform = make_platform()
        sampler = HardwareSampler(platform=platform, interval_s=60)
        sampler.start()
        assert sampler.snapshot().memory_available_bytes == 1_000_000
        sampler.stop()

    def test_start_calls_hardware_context_once(self):
        platform = make_platform()
        sampler = HardwareSampler(platform=platform, interval_s=60)
        sampler.start()
        platform.hardware_context.assert_called_once()
        sampler.stop()

    def test_stop_signals_thread(self):
        platform = make_platform()
        sampler = HardwareSampler(platform=platform, interval_s=60)
        sampler.start()
        sampler.stop()
        assert sampler.done.is_set()

    def test_snapshot_before_start_returns_empty_context(self):
        platform = make_platform()
        sampler = HardwareSampler(platform=platform, interval_s=60)
        assert sampler.snapshot() == HardwareContext()


class TestSamplerModule:
    def test_is_sampling_false_by_default(self):
        assert not is_sampling()

    def test_is_sampling_true_after_start(self):
        platform = make_platform()
        with patch("wildedge.platforms.CURRENT_PLATFORM", platform):
            start_sampler(interval_s=60)
        assert is_sampling()

    def test_is_sampling_false_after_stop(self):
        platform = make_platform()
        with patch("wildedge.platforms.CURRENT_PLATFORM", platform):
            start_sampler(interval_s=60)
        stop_sampler()
        assert not is_sampling()

    def test_stop_sampler_is_idempotent(self):
        stop_sampler()
        stop_sampler()  # should not raise

    def test_capture_hardware_uses_snapshot_when_sampling(self):
        platform = make_platform(HardwareContext(memory_available_bytes=42))
        with patch("wildedge.platforms.CURRENT_PLATFORM", platform):
            start_sampler(interval_s=60)
        ctx = capture_hardware()
        assert ctx.memory_available_bytes == 42

    def test_capture_hardware_live_read_when_not_sampling(self, monkeypatch):
        platform = make_platform(HardwareContext(memory_available_bytes=99))
        monkeypatch.setattr("wildedge.platforms.CURRENT_PLATFORM", platform)
        ctx = capture_hardware()
        assert ctx.memory_available_bytes == 99
        platform.hardware_context.assert_called_once()

    def test_capture_hardware_does_not_mutate_snapshot(self):
        platform = make_platform(HardwareContext(memory_available_bytes=1))
        with patch("wildedge.platforms.CURRENT_PLATFORM", platform):
            start_sampler(interval_s=60)
        from wildedge.platforms import _sampler

        original = _sampler.snapshot()
        capture_hardware(accelerator_actual="cuda")
        assert original.accelerator_actual is None

    def test_capture_hardware_accelerator_override(self):
        platform = make_platform()
        with patch("wildedge.platforms.CURRENT_PLATFORM", platform):
            start_sampler(interval_s=60)
        ctx = capture_hardware(accelerator_actual="mps")
        assert ctx.accelerator_actual == "mps"


class TestTrackInferenceHardware:
    def test_auto_attaches_when_sampling(self, monkeypatch):
        from wildedge.model import ModelHandle

        platform = make_platform(HardwareContext(memory_available_bytes=512))
        with patch("wildedge.platforms.CURRENT_PLATFORM", platform):
            start_sampler(interval_s=60)

        events = []
        handle = ModelHandle("m1", MagicMock(), events.append)
        handle.track_inference(duration_ms=10)
        assert events[0]["inference"]["hardware"]["memory_available_bytes"] == 512

    def test_no_hardware_when_not_sampling(self):
        from wildedge.model import ModelHandle

        events = []
        handle = ModelHandle("m1", MagicMock(), events.append)
        handle.track_inference(duration_ms=10)
        assert "hardware" not in events[0]["inference"]

    def test_explicit_hardware_takes_precedence_over_sampler(self):
        from wildedge.model import ModelHandle

        platform = make_platform(HardwareContext(memory_available_bytes=1))
        with patch("wildedge.platforms.CURRENT_PLATFORM", platform):
            start_sampler(interval_s=60)

        explicit = HardwareContext(memory_available_bytes=999)
        events = []
        handle = ModelHandle("m1", MagicMock(), events.append)
        handle.track_inference(duration_ms=10, hardware=explicit)
        assert events[0]["inference"]["hardware"]["memory_available_bytes"] == 999
