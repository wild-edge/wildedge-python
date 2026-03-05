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
) -> None:
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
            original_call = current_cls.__call__
            patched_call = make_patched_call(original_call)
            class_name = f"{current_cls.__name__}WildEdgePatched{patch_name.title()}"
            patched_cls = type(
                class_name,
                (current_cls,),
                {
                    "__call__": patched_call,
                    "__wildedge_patch_name__": patch_name,
                    "__wildedge_original_call__": original_call,
                },
            )
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
