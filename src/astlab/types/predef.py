from __future__ import annotations

__all__ = [
    "Predef",
    "get_predef",
]

from functools import cache, cached_property

from astlab.types.model import (
    ModuleInfo,
    NamedTypeInfo,
    builtins_module_info,
    ellipsis_type_info,
    none_type_info,
    typing_module_info,
)


@cache  # type: ignore[misc]
def get_predef() -> Predef:
    """Get predefined module & type infos."""
    return Predef()


class Predef:
    """Holds predefined module & type information. Mostly for `builtins` and `typing` modules."""

    @cached_property
    def typing_module(self) -> ModuleInfo:
        return typing_module_info()

    @cached_property
    def dataclasses_module(self) -> ModuleInfo:
        return ModuleInfo("dataclasses")

    @cached_property
    def builtins_module(self) -> ModuleInfo:
        return builtins_module_info()

    @cached_property
    def abc_module(self) -> ModuleInfo:
        return ModuleInfo("abc")

    @cached_property
    def contextlib_module(self) -> ModuleInfo:
        return ModuleInfo("contextlib")

    @cached_property
    def async_context_manager_decorator(self) -> NamedTypeInfo:
        return NamedTypeInfo("asynccontextmanager", self.contextlib_module)

    @cached_property
    def context_manager_decorator(self) -> NamedTypeInfo:
        return NamedTypeInfo("contextmanager", self.contextlib_module)

    @cached_property
    def object(self) -> NamedTypeInfo:
        return NamedTypeInfo("object", self.builtins_module)

    @cached_property
    def none_type(self) -> NamedTypeInfo:
        return none_type_info()

    @cached_property
    def ellipsis(self) -> NamedTypeInfo:
        return ellipsis_type_info()

    @cached_property
    def bool(self) -> NamedTypeInfo:
        return NamedTypeInfo("bool", self.builtins_module)

    @cached_property
    def int(self) -> NamedTypeInfo:
        return NamedTypeInfo("int", self.builtins_module)

    @cached_property
    def float(self) -> NamedTypeInfo:
        return NamedTypeInfo("float", self.builtins_module)

    @cached_property
    def complex(self) -> NamedTypeInfo:
        return NamedTypeInfo("complex", self.builtins_module)

    @cached_property
    def bytes(self) -> NamedTypeInfo:
        return NamedTypeInfo("bytes", self.builtins_module)

    @cached_property
    def bytearray(self) -> NamedTypeInfo:
        return NamedTypeInfo("bytearray", self.builtins_module)

    @cached_property
    def str(self) -> NamedTypeInfo:
        return NamedTypeInfo("str", self.builtins_module)

    @cached_property
    def list(self) -> NamedTypeInfo:
        return NamedTypeInfo("list", self.builtins_module)

    @cached_property
    def dict(self) -> NamedTypeInfo:
        return NamedTypeInfo("dict", self.builtins_module)

    @cached_property
    def set(self) -> NamedTypeInfo:
        return NamedTypeInfo("set", self.builtins_module)

    @cached_property
    def frozenset(self) -> NamedTypeInfo:
        return NamedTypeInfo("frozenset", self.builtins_module)

    @cached_property
    def property(self) -> NamedTypeInfo:
        return NamedTypeInfo("property", self.builtins_module)

    @cached_property
    def classmethod(self) -> NamedTypeInfo:
        return NamedTypeInfo("classmethod", self.builtins_module)

    @cached_property
    def abc_meta(self) -> NamedTypeInfo:
        return NamedTypeInfo("ABCMeta", self.abc_module)

    @cached_property
    def abstractmethod(self) -> NamedTypeInfo:
        return NamedTypeInfo("abstractmethod", self.abc_module)

    @cached_property
    def any(self) -> NamedTypeInfo:
        return NamedTypeInfo("Any", self.typing_module)

    @cached_property
    def generic(self) -> NamedTypeInfo:
        return NamedTypeInfo("Generic", self.typing_module)

    @cached_property
    def final_decorator(self) -> NamedTypeInfo:
        return NamedTypeInfo("final", self.typing_module)

    @cached_property
    def final(self) -> NamedTypeInfo:
        return NamedTypeInfo("Final", self.typing_module)

    @cached_property
    def class_var(self) -> NamedTypeInfo:
        return NamedTypeInfo("ClassVar", self.typing_module)

    @cached_property
    def type(self) -> NamedTypeInfo:
        return NamedTypeInfo("Type", self.typing_module)

    @cached_property
    def tuple(self) -> NamedTypeInfo:
        return NamedTypeInfo("Tuple", self.typing_module)

    @cached_property
    def container(self) -> NamedTypeInfo:
        return NamedTypeInfo("Container", self.typing_module)

    @cached_property
    def collection(self) -> NamedTypeInfo:
        return NamedTypeInfo("Collection", self.typing_module)

    @cached_property
    def sequence(self) -> NamedTypeInfo:
        return NamedTypeInfo("Sequence", self.typing_module)

    @cached_property
    def mutable_sequence(self) -> NamedTypeInfo:
        return NamedTypeInfo("MutableSequence", self.typing_module)

    @cached_property
    def dataclass_decorator(self) -> NamedTypeInfo:
        return NamedTypeInfo("dataclass", self.dataclasses_module)

    @cached_property
    def typed_dict(self) -> NamedTypeInfo:
        return NamedTypeInfo("TypedDict", self.typing_module)

    @cached_property
    def mapping(self) -> NamedTypeInfo:
        return NamedTypeInfo("Mapping", self.typing_module)

    @cached_property
    def mutable_mapping(self) -> NamedTypeInfo:
        return NamedTypeInfo("MutableMapping", self.typing_module)

    @cached_property
    def optional(self) -> NamedTypeInfo:
        return NamedTypeInfo("Optional", self.typing_module)

    @cached_property
    def union(self) -> NamedTypeInfo:
        return NamedTypeInfo("Union", self.typing_module)

    @cached_property
    def context_manager(self) -> NamedTypeInfo:
        return NamedTypeInfo("ContextManager", self.typing_module)

    @cached_property
    def async_context_manager(self) -> NamedTypeInfo:
        return NamedTypeInfo("AsyncContextManager", self.typing_module)

    @cached_property
    def iterator(self) -> NamedTypeInfo:
        return NamedTypeInfo("Iterator", self.typing_module)

    @cached_property
    def async_iterator(self) -> NamedTypeInfo:
        return NamedTypeInfo("AsyncIterator", self.typing_module)

    @cached_property
    def iterable(self) -> NamedTypeInfo:
        return NamedTypeInfo("Iterable", self.typing_module)

    @cached_property
    def async_iterable(self) -> NamedTypeInfo:
        return NamedTypeInfo("AsyncIterable", self.typing_module)

    @cached_property
    def literal(self) -> NamedTypeInfo:
        return NamedTypeInfo("Literal", self.typing_module)

    @cached_property
    def no_return(self) -> NamedTypeInfo:
        return NamedTypeInfo("NoReturn", self.typing_module)

    @cached_property
    def overload_decorator(self) -> NamedTypeInfo:
        return NamedTypeInfo("overload", self.typing_module)

    @cached_property
    def override_decorator(self) -> NamedTypeInfo:
        return NamedTypeInfo("override", self.typing_module)
