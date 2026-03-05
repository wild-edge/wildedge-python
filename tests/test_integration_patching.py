from __future__ import annotations

import types

from wildedge.integrations.gguf import GgufExtractor
from wildedge.integrations.hf import install_patch as install_hf_patch
from wildedge.integrations.onnx import OnnxExtractor
from wildedge.integrations.pytorch import PytorchExtractor


def test_hf_install_patch_is_idempotent(monkeypatch):
    import wildedge.integrations.hf as hf_mod

    def orig_hf_hub_download(repo_id, filename, **kwargs):
        return f"/tmp/{repo_id}/{filename}"

    fake_fd = types.SimpleNamespace(hf_hub_download=orig_hf_hub_download)
    fake_hf = types.SimpleNamespace(
        snapshot_download=lambda repo_id, **kwargs: "/tmp/s"
    )
    fake_consumer_module = types.SimpleNamespace(hf_hub_download=orig_hf_hub_download)

    monkeypatch.setattr(hf_mod, "_fd", fake_fd)
    monkeypatch.setattr(hf_mod, "_hf", fake_hf)
    monkeypatch.setattr(hf_mod, "_hf_hub_download_patched", False)
    monkeypatch.setattr(hf_mod, "_snapshot_download_patched", False)
    monkeypatch.setitem(
        hf_mod.sys.modules, "test_hf_consumer_mod", fake_consumer_module
    )

    install_hf_patch()
    first = fake_consumer_module.hf_hub_download
    install_hf_patch()
    second = fake_consumer_module.hf_hub_download
    assert first is second


def test_hf_install_patch_retries_unpatched_part(monkeypatch):
    import wildedge.integrations.hf as hf_mod

    monkeypatch.setattr(hf_mod, "_hf", object())
    monkeypatch.setattr(hf_mod, "_fd", object())
    monkeypatch.setattr(hf_mod, "_hf_hub_download_patched", False)
    monkeypatch.setattr(hf_mod, "_snapshot_download_patched", True)

    calls = {"hf": 0}

    def fake_install_hf():
        calls["hf"] += 1
        return calls["hf"] > 1

    monkeypatch.setattr(hf_mod, "_install_hf_hub_download_patch", fake_install_hf)
    install_hf_patch()
    assert hf_mod._hf_hub_download_patched is False
    install_hf_patch()
    assert hf_mod._hf_hub_download_patched is True
    assert calls["hf"] == 2


def test_onnx_install_auto_load_patch_is_idempotent(monkeypatch):
    import wildedge.integrations.onnx as onnx_mod

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

    fake_ort = types.SimpleNamespace(InferenceSession=FakeSession)
    monkeypatch.setattr(onnx_mod, "ort", fake_ort)
    monkeypatch.setattr(onnx_mod, "_ort_patched", False)

    def client_ref():
        return None

    OnnxExtractor.install_auto_load_patch(client_ref)
    first_cls = fake_ort.InferenceSession
    OnnxExtractor.install_auto_load_patch(client_ref)
    second_cls = fake_ort.InferenceSession
    assert first_cls is second_cls


def test_timm_install_patch_is_idempotent(monkeypatch):
    import wildedge.integrations.pytorch as torch_mod

    def create_model(*args, **kwargs):
        return object()

    fake_timm = types.SimpleNamespace(create_model=create_model)
    monkeypatch.setattr(torch_mod, "_timm", fake_timm)
    monkeypatch.setattr(torch_mod, "_timm_patched", False)

    def client_ref():
        return None

    PytorchExtractor.install_timm_patch(client_ref)
    first = fake_timm.create_model
    PytorchExtractor.install_timm_patch(client_ref)
    second = fake_timm.create_model
    assert first is second


def test_gguf_install_auto_load_patch_is_idempotent(monkeypatch):
    import wildedge.integrations.gguf as gguf_mod

    class FakeLlama:
        def __init__(self, *args, **kwargs):
            pass

    fake_llama_cpp = types.SimpleNamespace(Llama=FakeLlama)
    monkeypatch.setattr(gguf_mod, "_llama_cpp", fake_llama_cpp)
    monkeypatch.setattr(gguf_mod, "_llama_patched", False)

    def client_ref():
        return None

    GgufExtractor.install_auto_load_patch(client_ref)
    first = fake_llama_cpp.Llama.__init__
    GgufExtractor.install_auto_load_patch(client_ref)
    second = fake_llama_cpp.Llama.__init__
    assert first is second
