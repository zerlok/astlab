from __future__ import annotations

import typing as t
import weakref
from functools import lru_cache, wraps

F = t.TypeVar("F")


def lru_cache_method(max_size: t.Optional[int] = None) -> t.Callable[[F], F]:
    """
    A workaround for `lru_cache` decorator for instance methods.

    Builtin `functools.lru_cache` implementation is bad for instance methods, due memory leaks, see ruff B019.

    Weak ref to self argument helps to avoid memory leaks.
    """

    def inner(func: F) -> F:
        assert callable(func)

        @lru_cache(maxsize=max_size)  # type: ignore[misc]
        def patch_weak_ref(self: weakref.ReferenceType[object], *args: object, **kwargs: object) -> object:
            return func(self(), *args, **kwargs)  # type: ignore[misc]

        @wraps(func)  # type: ignore[misc]
        def wrapper(self: object, *args: object, **kwargs: object) -> object:  # type: ignore[misc]
            return patch_weak_ref(weakref.ref(self), *args, **kwargs)

        wrapper.cache_info = patch_weak_ref.cache_info  # type: ignore[attr-defined,misc]
        wrapper.cache_clear = patch_weak_ref.cache_clear  # type: ignore[attr-defined,misc]

        return t.cast("F", wrapper)

    return inner
