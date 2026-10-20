from __future__ import annotations

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

import wildedge
import wildedge.convenience as convenience
import wildedge.hubs.huggingface as huggingface
import wildedge.instrumentation as instrumentation
from wildedge.client import WildEdge
from wildedge.defaults import set_default_client
from wildedge.instrumentation import BoundClientReference, claim_instrumentation
from wildedge.platforms.device_info import DeviceInfo


class Owner:
    closed = False


def test_sequential_replacement_and_opt_in():
    first = Owner()
    route = claim_instrumentation(first, "onnx")
    assert route() is first
    assert claim_instrumentation(first, "onnx")() is first
    first.closed = True
    assert route() is None
    second = Owner()
    other = claim_instrumentation(second, "openai")
    assert other() is second
    assert route() is None
    claim_instrumentation(second, "onnx")
    assert route() is second


def test_live_owner_conflict_does_not_redirect():
    first, second = Owner(), Owner()
    route = claim_instrumentation(first, "onnx")
    with pytest.raises(RuntimeError, match="another live"):
        claim_instrumentation(second, "openai")
    assert route() is first


def test_dead_owner_can_be_replaced():
    first = Owner()
    ref = weakref.ref(first)
    route = claim_instrumentation(first, "onnx")
    del first
    gc.collect()
    assert ref() is None
    second = Owner()
    claim_instrumentation(second, "onnx")
    assert route() is second


def test_provider_reference_never_follows_replacement(caplog):
    caplog.set_level("WARNING", logger="wildedge")
    first, second = Owner(), Owner()
    claim_instrumentation(first, "openai")
    ref = BoundClientReference(first)
    first.closed = True
    claim_instrumentation(second, "openai")
    assert ref() is None
    assert ref() is None
    assert len([r for r in caplog.records if "recreate" in r.message]) == 1


@pytest.fixture
def clients():
    created = []
    with (
        patch(
            "wildedge.client.detect_device",
            return_value=DeviceInfo(device_id="replacement-test", device_type="linux"),
        ),
        patch("wildedge.client.Consumer"),
        patch("wildedge.client.Transmitter"),
    ):

        def create():
            client = WildEdge(
                dsn="https://test@localhost/test",
                sampling_interval_s=0,
                enable_offline_persistence=False,
            )
            created.append(client)
            return client

        yield create
    for client in created:
        client.close()


@pytest.mark.parametrize("integration", sorted(WildEdge.PATCH_INSTALLERS))
def test_shared_route_reuses_first_factory_binding(clients, monkeypatch, integration):
    routes = []
    installer = MagicMock(
        side_effect=lambda route: routes.append(route) if not routes else None
    )
    monkeypatch.setitem(WildEdge.PATCH_INSTALLERS, integration, installer)
    first = clients()
    first.instrument(integration)
    assert routes[0]() is first
    first.close()
    second = clients()
    second.instrument(integration)
    assert routes[0]() is second
    assert len(routes) == 1


def test_closed_model_handle_does_not_move_projects(clients, caplog):
    caplog.set_level("WARNING", logger="wildedge")
    first = clients()
    handle = first.register_model(object(), model_id="model", auto_instrument=False)
    first.close()
    second = clients()
    handle.track_inference(duration_ms=1)
    handle.track_inference(duration_ms=2)
    assert first.pending_count() == second.pending_count() == 0
    assert len([r for r in caplog.records if "closed SDK client" in r.message]) == 1


def test_init_replaces_closed_default(clients, monkeypatch):
    first = clients()
    set_default_client(first)
    first.close()
    second = clients()
    monkeypatch.setattr(convenience, "WildEdge", lambda **kwargs: second)
    assert wildedge.init() is second


def test_close_is_idempotent(clients):
    client = clients()
    client.close()
    client.close()
    client.consumer.close.assert_called_once()


def test_init_conflict_cleans_up_new_client_without_stopping_owner_sampler(
    clients, monkeypatch
):
    first = clients()
    claim_instrumentation(first, "onnx")
    set_default_client(first)
    second = clients()
    monkeypatch.setattr(convenience, "WildEdge", lambda **kwargs: second)
    with patch("wildedge.client.stop_sampler") as stop:
        with pytest.raises(RuntimeError, match="another live"):
            wildedge.init(dsn="https://other@localhost/other", integrations=["onnx"])
        stop.assert_not_called()
    assert second.closed
    assert not first.closed
    assert convenience.peek_default_client() is first


@pytest.mark.parametrize("hub", ["huggingface", "torchhub"])
def test_hub_factory_route_follows_replacement(clients, monkeypatch, hub):
    routes = []

    class Tracker:
        def can_install(self):
            return True

        def install_patch(self, route):
            if not routes:
                routes.append(route)

    monkeypatch.setitem(WildEdge.HUB_TRACKER_CLASSES, hub, Tracker)
    first = clients()
    first.instrument(None, hubs=[hub])
    assert routes[0]() is first
    first.close()
    second = clients()
    second.instrument(None, hubs=[hub])
    assert routes[0]() is second
    assert first.hub_trackers[hub] is not second.hub_trackers[hub]


def test_changing_default_does_not_change_instrumentation_owner(clients):
    first, second = clients(), clients()
    route = claim_instrumentation(first, "openai")
    set_default_client(second)
    assert route() is first


def test_closed_client_cannot_instrument(clients):
    client = clients()
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.instrument("onnx")


def test_fork_child_replaces_inherited_lock():
    first = Owner()
    route = claim_instrumentation(first, "onnx")
    with instrumentation._lock:
        instrumentation._reset_lock_after_fork()
        assert route() is first


def test_concurrent_claims_choose_one_owner():
    owners = [Owner(), Owner()]
    barrier = threading.Barrier(2, timeout=2)

    def acquire(owner):
        barrier.wait()
        try:
            return claim_instrumentation(owner, "onnx")
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(acquire, owners))
    routes = [route for route in results if route is not None]
    assert len(routes) == 1
    assert routes[0]() in owners


def test_hub_provenance_buffer_does_not_cross_owners(clients, monkeypatch):
    first = clients()
    route = claim_instrumentation(first, "hub:huggingface")
    monkeypatch.setattr(huggingface, "_client_route", route)
    monkeypatch.setattr(huggingface, "_local", threading.local())
    record = {
        "repo_id": "first-project/model",
        "size": 1,
        "duration_ms": 1,
        "cache_hit": True,
        "bandwidth_bps": None,
    }
    huggingface._record_download(**record)
    first.close()
    second = clients()
    claim_instrumentation(second, "onnx")
    huggingface._record_download(**record)
    claim_instrumentation(second, "hub:huggingface")
    assert huggingface.HuggingFaceHubTracker().drain() == []
    huggingface._record_download(**{**record, "repo_id": "second-project/model"})
    assert [r["repo_id"] for r in huggingface.HuggingFaceHubTracker().drain()] == [
        "second-project/model"
    ]
