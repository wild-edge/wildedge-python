from __future__ import annotations

import threading
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wildedge.model import ModelHandle, ModelInfo

_CALL_PATCH_LOCK = threading.Lock()
_PATCHED_CALL_CLASSES: weakref.WeakKeyDictionary[type, dict[str, type]] = (
    weakref.WeakKeyDictionary()
)


def patch_instance_call_once(
    obj: object,
    *,
    patch_name: str,
    make_patched_call: Callable[[Callable[..., Any]], Callable[..., Any]],
    method: str = "__call__",
    extra_methods: dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]]
    | None = None,
) -> None:
    """Swap obj's class for a cached subclass whose ``method`` is wrapped.

    ``method`` defaults to ``__call__``; pass another name when the runtime's
    entry points all funnel through a different method. ``extra_methods`` maps
    more method names to wrapper factories, patched in the same subclass so a
    repeat call with the same ``patch_name`` stays a no-op.
    """
    current_cls = type(obj)
    if getattr(current_cls, "__wildedge_patch_name__", None) == patch_name:
        return

    with _CALL_PATCH_LOCK:
        current_cls = type(obj)
        if getattr(current_cls, "__wildedge_patch_name__", None) == patch_name:
            return

        per_class = _PATCHED_CALL_CLASSES.get(current_cls)
        if per_class is None:
            per_class = {}
            _PATCHED_CALL_CLASSES[current_cls] = per_class

        patched_cls = per_class.get(patch_name)
        if patched_cls is None:
            original_call = getattr(current_cls, method)
            patched_call = make_patched_call(original_call)
            namespace: dict[str, Any] = {
                method: patched_call,
                "__wildedge_patch_name__": patch_name,
                "__wildedge_original_call__": original_call,
            }
            for extra_name, make_extra in (extra_methods or {}).items():
                original_extra = getattr(current_cls, extra_name, None)
                if original_extra is not None:
                    namespace[extra_name] = make_extra(original_extra)
            class_name = f"{current_cls.__name__}WildEdgePatched{patch_name.title()}"
            patched_cls = type(class_name, (current_cls,), namespace)
            per_class[patch_name] = patched_cls

        obj.__class__ = patched_cls  # type: ignore[assignment]


class BaseExtractor(ABC):
    @abstractmethod
    def can_handle(self, obj: object) -> bool:
        """Return True if this extractor knows how to handle obj."""
        ...

    @abstractmethod
    def extract_info(
        self, obj: object, overrides: dict
    ) -> tuple[str | None, ModelInfo]:
        """
        Extract model metadata from obj, applying user overrides on top.

        Returns (model_id, ModelInfo). model_id=None means the runtime has no
        reliable stable name and the caller must require an explicit id from the user.
        """
        ...

    @abstractmethod
    def install_hooks(self, obj: object, handle: ModelHandle) -> None:
        """Patch obj in-place to auto-track inference calls via handle."""
        ...

    def memory_bytes(self, obj: object) -> int | None:
        """Best-effort estimate of model memory footprint in bytes."""
        return None
