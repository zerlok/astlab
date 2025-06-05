from __future__ import annotations

__all__ = [
    "ModuleLoader",
    "TypeLoader",
]

import importlib
import typing as t
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
    @lru_cache_method()
    def load(self, info: t.Union[PackageInfo, ModuleInfo, Path]) -> ModuleType:
        return import_module_path(info) if isinstance(info, Path) else importlib.import_module(info.qualname)


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
            return type_[*type_params] if len(type_params) > 1 else type_[type_params[0]]  # type: ignore[index,misc]

        elif isinstance(info, LiteralTypeInfo):
            return t.Literal[*info.values]

        else:
            assert_never(info)  # noqa: RET503
