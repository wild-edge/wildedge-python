from __future__ import annotations

import pytest


def test_ultralytics_import_and_instrument(compat_client):
    ultralytics = pytest.importorskip("ultralytics")
    assert getattr(ultralytics, "__version__", None)
    compat_client.instrument("ultralytics")
