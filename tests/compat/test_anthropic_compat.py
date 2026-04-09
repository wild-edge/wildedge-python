from __future__ import annotations

import pytest


def test_anthropic_import_and_instrument(compat_client):
    anthropic = pytest.importorskip("anthropic")
    assert getattr(anthropic, "__version__", None)
    compat_client.instrument("anthropic")
