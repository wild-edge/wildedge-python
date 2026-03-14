from __future__ import annotations

from unittest.mock import patch

from wildedge.client import WildEdge
from wildedge.device import DeviceInfo


class _DummyConsumer:
    def __init__(self, *args, **kwargs):
        pass

    def flush(self, timeout: float = 5.0) -> None:
        pass

    def close(self, timeout: float | None = None) -> None:
        pass

    def _pause(self) -> None:
        pass

    def _resume(self) -> None:
        pass


class _Model:
    pass


def test_offline_replay_restores_model_registry_for_pending_events(tmp_path):
    queue_dir = tmp_path / "queue"
    dead_dir = tmp_path / "dead"
    with (
        patch(
            "wildedge.client.detect_device",
            return_value=DeviceInfo(device_id="d", device_type="linux"),
        ),
        patch("wildedge.client.Transmitter"),
        patch("wildedge.client.Consumer", _DummyConsumer),
    ):
        client_a = WildEdge(
            dsn="https://secret@ingest.wildedge.dev/proj",
            app_identity="app-a",
            offline_queue_dir=str(queue_dir),
            dead_letter_dir=str(dead_dir),
            enable_offline_persistence=True,
        )
        client_a.register_model(
            _Model(),
            model_id="ResNet",
            source="local",
            family="resnet",
            version="1.0",
            quantization="fp32",
        )
        client_a.publish(
            {"event_id": "e1", "event_type": "model_load", "model_id": "ResNet"}
        )
        client_a.close()

        client_b = WildEdge(
            dsn="https://secret@ingest.wildedge.dev/proj",
            app_identity="app-a",
            offline_queue_dir=str(queue_dir),
            dead_letter_dir=str(dead_dir),
            enable_offline_persistence=True,
        )

    assert client_b.queue.length() == 1
    models = client_b.registry.snapshot()
    assert "ResNet" in models
    assert models["ResNet"]["model_name"] == "_Model"
