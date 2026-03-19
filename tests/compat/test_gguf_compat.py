from __future__ import annotations

import pytest


def test_gguf_import_and_instrument(compat_client):
    llama_cpp = pytest.importorskip("llama_cpp")
    assert hasattr(llama_cpp, "Llama")
    compat_client.instrument("gguf")
