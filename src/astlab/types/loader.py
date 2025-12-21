from __future__ import annotations

__all__ = [
    "ModuleLoader",
    "TypeLoader",
    "TypeLoaderError",
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
    EnumTypeInfo,
    LiteralTypeInfo,
    ModuleInfo,
    NamedTypeInfo,
    PackageInfo,
    RuntimeType,
    TypeInfo,
    TypeVarInfo,
    UnionTypeInfo,
    ellipsis_type_info,
    none_type_info,
)
from astlab.version import PythonVersion

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


class TypeLoaderError(Exception):
    pass


class TypeLoader:
    """Loads runtime type from provided info."""

    def __init__(
        self,
        module: t.Optional[ModuleLoader] = None,
        python_version: t.Optional[PythonVersion] = None,
    ) -> None:
        self.__module = module or ModuleLoader()
        self.__version = PythonVersion.get(python_version)

    @lru_cache_method()
    def load(self, info: TypeInfo) -> RuntimeType:
        if isinstance(info, ModuleInfo):
            rtt = self.__load_module(info)

        elif isinstance(info, TypeVarInfo):
            rtt = self.__load_type_var(info)

        elif isinstance(info, NamedTypeInfo):
            if info == none_type_info():
                rtt = None

            elif info == ellipsis_type_info():
                rtt = Ellipsis

            else:
                rtt = self.__load_named_type(info)

        elif isinstance(info, LiteralTypeInfo):
            rtt = getitem(t.Literal, info.values)

        elif isinstance(info, EnumTypeInfo):
            rtt = self.__load_type_by_name(info)

        elif isinstance(info, UnionTypeInfo):
            rtt = self.__load_union_type(info)

        else:
            assert_never(info)

        return rtt

    def clear_cache(self) -> None:
        self.load.cache_clear()  # type: ignore[attr-defined]
        self.__module.clear_cache()

    def __load_module(self, info: ModuleInfo) -> RuntimeType:
        try:
            return self.__module.load(info)

        except ImportError as err:
            msg = "module can't be loaded"
            raise TypeLoaderError(msg, info) from err

    def __load_type_var(self, info: TypeVarInfo) -> RuntimeType:
        # noinspection PyTypeHints
        return t.TypeVar(
            name=info.name,
            bound=self.load(info.lower) if info.lower else None,
            covariant=info.variance == "covariant",
            contravariant=info.variance == "contravariant",
        )

    def __load_named_type(self, info: NamedTypeInfo) -> RuntimeType:
        rtt = self.__load_type_by_name(info)
        return self.__parametrize_type(info, rtt, info.type_params)

    def __load_type_by_name(self, info: t.Union[NamedTypeInfo, EnumTypeInfo]) -> RuntimeType:
        container: object = self.load(info.module)
        try:
            for name in info.namespace:
                container = getattr(container, name)

            # NOTE: need to check that we loaded a type.
            return getattr(container, info.name)  # type: ignore[misc]

        except AttributeError as err:
            msg = "module has not attribute"
            raise TypeLoaderError(msg, info) from err

    if sys.version_info >= (3, 10):

        def __load_union_type(self, info: UnionTypeInfo) -> RuntimeType:
            if self.__version < PythonVersion.PY310:
                return self.__load_typing_union(info)

            head, *tail = info.values

            rtt = self.load(head)
            for val in tail:
                rtt |= self.load(val)

            return rtt

    else:

        def __load_union_type(self, info: UnionTypeInfo) -> RuntimeType:
            return self.__load_typing_union(info)

    def __load_typing_union(self, info: UnionTypeInfo) -> RuntimeType:
        return self.__parametrize_type(info, t.Union, info.values)

    def __parametrize_type(self, info: TypeInfo, rtt: RuntimeType, params: t.Sequence[TypeInfo]) -> RuntimeType:
        if not params:
            return rtt

        # TODO: fix recursive type load
        loaded_type_params = tuple(self.load(param) for param in params)

        try:
            return (
                getitem(rtt, loaded_type_params) if len(loaded_type_params) > 1 else getitem(rtt, loaded_type_params[0])  # type: ignore[arg-type,misc]
            )

        except TypeError as err:
            msg = "type params can't be applied to type"
            raise TypeLoaderError(msg, info) from err
