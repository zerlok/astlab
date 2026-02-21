# NOTE: it's hard to avoid `Any` in inspector code.
# mypy: disable-error-code="misc"

from __future__ import annotations

__all__ = [
    "TypeInspector",
]

import enum
import sys
import types
import typing as t

from astlab.cache import lru_cache_method
from astlab.types.model import (
    EnumTypeInfo,
    EnumTypeValue,
    LiteralTypeInfo,
    ModuleInfo,
    NamedTypeInfo,
    RuntimeType,
    TypeInfo,
    TypeVarInfo,
    UnionTypeInfo,
    ellipsis_type_info,
    none_type_info,
    typing_module_info,
)
from astlab.version import PythonVersion

if sys.version_info >= (3, 10):
    UnionType = types.UnionType

else:

    class UnionType:
        pass


class TypeInspector:
    """Provides type info from runtime type."""

    def __init__(self, python_version: t.Optional[PythonVersion] = None) -> None:
        self.__version = PythonVersion.parse(python_version)

    @lru_cache_method()
    def inspect(self, type_: RuntimeType) -> TypeInfo:
        if type_ is None:
            return none_type_info()

        elif type_ is Ellipsis:
            return ellipsis_type_info()

        elif isinstance(
            type_,
            t._LiteralGenericAlias,  # type: ignore[attr-defined] # noqa: SLF001
        ):
            return self.__inspect_literal_type(type_)

        elif isinstance(type_, UnionType) or t.get_origin(type_) is t.Union:
            return self.__inspect_union_type(type_)

        else:
            return self.__inspect_named_type(type_)

    def __inspect_literal_type(self, type_: RuntimeType) -> LiteralTypeInfo:
        values = t.get_args(type_)

        if not values or not all(val is not None or isinstance(val, (bool, int, bytes, str)) for val in values):
            msg = "invalid literal type"
            raise TypeError(msg, type_)

        return LiteralTypeInfo(values=values)

    def __inspect_union_type(self, type_: RuntimeType) -> TypeInfo:
        type_params = self.__unpack_type_params(type_)
        values = tuple(self.inspect(param) for param in type_params)

        return (
            UnionTypeInfo(values=values)
            # if self.__version >= PythonVersion.PY310 or len(values) != 2 or values[1] != get_predef().none
            # else replace(get_predef().optional, type_params=values[:1])
        )

    def __inspect_named_type(self, type_: RuntimeType) -> TypeInfo:
        type_params = self.__unpack_type_params(type_)
        module, namespace, name = self.__get_module_naming(type_)

        if isinstance(type_, t.TypeVar):
            return TypeVarInfo(
                name=type_.__name__,
                module=module,
                namespace=namespace,
                variance=(
                    "covariant" if type_.__covariant__ else "contravariant" if type_.__contravariant__ else "invariant"
                ),
                constraints=tuple(self.inspect(co) for co in type_.__constraints__),
                lower=self.inspect(type_.__bound__) if type_.__bound__ is not None else None,
            )

        elif isinstance(type_, type) and issubclass(type_, enum.Enum):
            return EnumTypeInfo(
                name=name,
                module=module,
                namespace=tuple(namespace),
                values=tuple(EnumTypeValue(name=enum_value.name, value=enum_value.value) for enum_value in type_),
            )

        else:
            return NamedTypeInfo(
                name=name,
                module=module,
                namespace=tuple(namespace),
                type_params=tuple(self.inspect(param) for param in type_params),
            )

    if sys.version_info >= (3, 11):

        def __get_module_naming(self, type_: RuntimeType) -> tuple[ModuleInfo, t.Sequence[str], str]:
            module = ModuleInfo.from_str(type_.__module__)
            qualname = getattr(type_, "__qualname__", getattr(type_, "__name__", None)) or repr(type_)
            *namespace, name = qualname.split(".")
            return module, namespace, name

    else:

        def __get_module_naming(self, type_: RuntimeType) -> tuple[ModuleInfo, t.Sequence[str], str]:
            module = ModuleInfo.from_str(type_.__module__)

            if module == typing_module_info():
                origin = t.get_origin(type_) or type_
                supertype = getattr(origin, "__supertype__", None)

                if supertype is not None:
                    msg = "can't get module naming for NewType"
                    raise TypeError(msg, getattr(type_, "__name__", type_), supertype)

                fullname = str(type_)
                sq_bracket_idx = fullname.find("[")
                qualname = fullname[
                    len(type_.__module__) + 1 : sq_bracket_idx if sq_bracket_idx >= 0 else len(fullname)
                ]
                *namespace, name = qualname.split(".")

            else:
                try:
                    *namespace, name = type_.__qualname__.split(".")  # type: ignore[union-attr]

                except AttributeError:
                    origin = t.get_origin(type_)
                    if origin is None:
                        msg = "can't get module naming for type"
                        raise TypeError(msg, type_) from None

                    *namespace, name = origin.__qualname__.split(".")

            return module, namespace, name

    def __unpack_type_params(self, type_: RuntimeType) -> t.Sequence[RuntimeType]:
        args: t.Optional[t.Sequence[RuntimeType]] = t.get_args(type_)
        return args or ()
