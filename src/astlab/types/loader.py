from __future__ import annotations

__all__ = [
    "ModuleLoader",
    "TypeLoader",
]

import importlib
import sys
import typing as t
from contextlib import contextmanager
from operator import getitem
from pathlib import Path

from astlab._typing import assert_never
from astlab.cache import lru_cache_method
from astlab.reader import import_module_path
from astlab.types.model import (
    LiteralTypeInfo,
    ModuleInfo,
    NamedTypeInfo,
    PackageInfo,
    RuntimeType,
    TypeInfo,
    ellipsis_type_info,
    none_type_info,
)

if t.TYPE_CHECKING:
    from types import ModuleType


class ModuleLoader:
    @classmethod
    @contextmanager
    def with_sys_path(cls, *paths: Path) -> t.Iterator[ModuleLoader]:
        sys.path.extend(str(path) for path in paths)
        loader = cls()
        try:
            yield loader

        finally:
            loader.clear_cache()

            for path in paths:
                sys.path.remove(str(path))

    @lru_cache_method()
    def load(self, info: t.Union[PackageInfo, ModuleInfo, Path]) -> ModuleType:
        return import_module_path(info) if isinstance(info, Path) else importlib.import_module(info.qualname)

    def clear_cache(self) -> None:
        self.load.cache_clear()  # type: ignore[attr-defined]
        importlib.invalidate_caches()


class TypeLoader:
    """Loads runtime type from provided info."""

    def __init__(self, module: t.Optional[ModuleLoader] = None) -> None:
        self.__module = module or ModuleLoader()

    @lru_cache_method()
    def load(self, info: TypeInfo) -> RuntimeType:
        if isinstance(info, NamedTypeInfo):
            if info == none_type_info():
                return None

            elif info == ellipsis_type_info():
                return Ellipsis

            value: object = self.__module.load(info.module)

            for name in info.namespace:
                value = getattr(value, name)

            # NOTE: need to check that we loaded a type.
            type_: object = getattr(value, info.name)

            if not info.type_params:
                return type_

            # TODO: fix recursive type
            type_params = tuple(self.load(tp) for tp in info.type_params)
            return getitem(type_, type_params) if len(type_params) > 1 else getitem(type_, type_params[0])  # type: ignore[call-overload, misc]

        elif isinstance(info, LiteralTypeInfo):
            return getitem(t.Literal, info.values)

        else:
            assert_never(info)
