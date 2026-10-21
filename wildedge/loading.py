"""Import optional parts of the SDK only when a feature uses them.

Importing ``wildedge`` must not import ML frameworks or API clients. The
registries name each adapter by import path instead, and these helpers load it
on first use.
"""

from __future__ import annotations

from functools import cache
from importlib import import_module
from types import ModuleType
from typing import Any


@cache
def resolve(target: str) -> Any:
    """Import ``package.module:name`` and return that attribute."""
    module, name = target.split(":", 1)
    return getattr(import_module(module), name)


def lazy_callable(target: str):
    """A function that imports ``target`` on its first call and forwards to it."""

    def call(*args, **kwargs):
        return resolve(target)(*args, **kwargs)

    return call


def optional_module(name: str) -> ModuleType | None:
    """Import an optional third-party runtime, or None if it is not installed."""
    try:
        return import_module(name)
    except ImportError:
        return None
