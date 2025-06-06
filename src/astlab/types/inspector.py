# NOTE: it's hard to avoid `Any` in inspector code.
# mypy: disable-error-code="misc"

from __future__ import annotations

__all__ = [
    "TypeInspector",
]

import typing as t

from astlab.cache import lru_cache_method
from astlab.types.model import LiteralTypeInfo, ModuleInfo, NamedTypeInfo, RuntimeType, TypeInfo


class TypeInspector:
    """Provides type info from runtime type."""

    @lru_cache_method()
    def inspect(self, type_: RuntimeType) -> TypeInfo:
        if isinstance(
            type_,
            t._LiteralGenericAlias,  # type: ignore[attr-defined,misc] # noqa: SLF001
        ):
            args = t.get_args(type_)
            if not all(arg is not None or isinstance(arg, (bool, int, bytes, str)) for arg in args):
                msg = "invalid literal type"
                raise TypeError(msg, type_)

            return LiteralTypeInfo(values=args or ())

        module = ModuleInfo.from_str(type_.__module__)
        # TODO: check if all types actually have `__qualname__`
        *namespace, name = type_.__qualname__.split(".")  # type: ignore[union-attr]

        return NamedTypeInfo(
            name=name,
            module=module,
            namespace=tuple(namespace),
            # TODO: fix recursive type
            type_params=tuple(self.inspect(param) for param in self.__get_type_params(type_)),
        )

    def __get_type_params(self, type_: RuntimeType) -> t.Sequence[RuntimeType]:
        origin: t.Optional[RuntimeType] = t.get_origin(type_)
        args: t.Optional[t.Sequence[RuntimeType]] = t.get_args(type_)

        # patch Union[T, None] => Optional[T]
        if origin is t.Union and args is not None and len(args) == 2 and args[1] is type(None):  # noqa: PLR2004
            return args[:1]

        return args or ()
