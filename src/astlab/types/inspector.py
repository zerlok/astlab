# NOTE: it's hard to avoid `Any` in inspector code.
# mypy: disable-error-code="misc"

from __future__ import annotations

__all__ = [
    "TypeInspector",
]

import enum
import sys
import typing as t
from dataclasses import replace

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
    typing_module_info,
)
from astlab.types.predef import get_predef


class TypeInspector:
    """Provides type info from runtime type."""

    @lru_cache_method()
    def inspect(self, type_: t.Union[TypeInfo, RuntimeType]) -> TypeInfo:
        if isinstance(type_, (ModuleInfo, TypeVarInfo, NamedTypeInfo, LiteralTypeInfo, EnumTypeInfo)):
            return type_

        if isinstance(
            type_,
            t._LiteralGenericAlias,  # type: ignore[attr-defined] # noqa: SLF001
        ):
            args = t.get_args(type_)
            if not all(arg is not None or isinstance(arg, (bool, int, bytes, str)) for arg in args):
                msg = "invalid literal type"
                raise TypeError(msg, type_)

            return LiteralTypeInfo(values=args or ())

        origin, type_params = self.__unpack_generic(type_)
        module, namespace, name = self.__get_module_naming(origin)

        if isinstance(origin, t.TypeVar):
            return TypeVarInfo(
                name=origin.__name__,
                module=module,
                namespace=namespace,
                variance=(
                    "covariant"
                    if origin.__covariant__
                    else "contravariant"
                    if origin.__contravariant__
                    else "invariant"
                ),
                lower=self.inspect(origin.__bound__)
                if origin.__bound__ is not None
                else replace(get_predef().union, type_params=tuple(self.inspect(co) for co in origin.__constraints__))
                if origin.__constraints__
                else None,
            )

        if isinstance(origin, type) and issubclass(origin, enum.Enum):
            return EnumTypeInfo(
                name=name,
                module=module,
                namespace=tuple(namespace),
                values=tuple(EnumTypeValue(name=enum_value.name, value=enum_value.value) for enum_value in origin),
            )

        return NamedTypeInfo(
            name=name,
            module=module,
            namespace=tuple(namespace),
            type_params=tuple(self.inspect(type_param) for type_param in type_params),
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

    def __unpack_generic(self, type_: RuntimeType) -> tuple[RuntimeType, t.Sequence[RuntimeType]]:
        origin: t.Optional[RuntimeType] = t.get_origin(type_)
        args: t.Optional[t.Sequence[RuntimeType]] = t.get_args(type_)

        # patch Union[T, None] => Optional[T]
        if origin is t.Union and args is not None and len(args) == 2 and args[1] is type(None):  # noqa: PLR2004
            return t.Optional, args[:1]

        return type_, args or ()
