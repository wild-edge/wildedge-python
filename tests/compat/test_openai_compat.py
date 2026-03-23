from __future__ import annotations

import pytest


def test_openai_import_and_instrument(compat_client):
    openai = pytest.importorskip("openai")
    assert getattr(openai, "__version__", None)
    compat_client.instrument("openai")
