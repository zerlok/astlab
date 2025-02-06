from __future__ import annotations

__all__ = [
    "get_predefs",
]

from functools import cache, cached_property

from astlab.info import ModuleInfo, TypeInfo


@cache  # type: ignore[misc]
def get_predefs() -> Predefs:
    return Predefs()


class Predefs:
    @cached_property
    def typing_module(self) -> ModuleInfo:
        return ModuleInfo(None, "typing")

    @cached_property
    def dataclasses_module(self) -> ModuleInfo:
        return ModuleInfo(None, "dataclasses")

    @cached_property
    def builtins_module(self) -> ModuleInfo:
        return ModuleInfo(None, "builtins")

    @cached_property
    def abc_module(self) -> ModuleInfo:
        return ModuleInfo(None, "abc")

    @cached_property
    def contextlib_module(self) -> ModuleInfo:
        return ModuleInfo(None, "contextlib")

    @cached_property
    def async_context_manager_decorator(self) -> TypeInfo:
        return TypeInfo("asynccontextmanager", self.contextlib_module)

    @cached_property
    def context_manager_decorator(self) -> TypeInfo:
        return TypeInfo("contextmanager", self.contextlib_module)

    @cached_property
    def none_type(self) -> TypeInfo:
        return TypeInfo("NoneType", self.builtins_module)

    @cached_property
    def bool(self) -> TypeInfo:
        return TypeInfo("bool", self.builtins_module)

    @cached_property
    def int(self) -> TypeInfo:
        return TypeInfo("int", self.builtins_module)

    @cached_property
    def float(self) -> TypeInfo:
        return TypeInfo("float", self.builtins_module)

    @cached_property
    def str(self) -> TypeInfo:
        return TypeInfo("str", self.builtins_module)

    @cached_property
    def list(self) -> TypeInfo:
        return TypeInfo("list", self.builtins_module)

    @cached_property
    def dict(self) -> TypeInfo:
        return TypeInfo("dict", self.builtins_module)

    @cached_property
    def set(self) -> TypeInfo:
        return TypeInfo("set", self.builtins_module)

    @cached_property
    def property(self) -> TypeInfo:
        return TypeInfo("property", self.builtins_module)

    @cached_property
    def classmethod(self) -> TypeInfo:
        return TypeInfo("classmethod", self.builtins_module)

    @cached_property
    def abc_meta(self) -> TypeInfo:
        return TypeInfo("ABCMeta", self.abc_module)

    @cached_property
    def abstractmethod(self) -> TypeInfo:
        return TypeInfo("abstractmethod", self.abc_module)

    @cached_property
    def generic(self) -> TypeInfo:
        return TypeInfo("Generic", self.typing_module)

    @cached_property
    def final_decorator(self) -> TypeInfo:
        return TypeInfo("final", self.typing_module)

    @cached_property
    def final(self) -> TypeInfo:
        return TypeInfo("Final", self.typing_module)

    @cached_property
    def class_var(self) -> TypeInfo:
        return TypeInfo("ClassVar", self.typing_module)

    @cached_property
    def type(self) -> TypeInfo:
        return TypeInfo("Type", self.typing_module)

    @cached_property
    def tuple(self) -> TypeInfo:
        return TypeInfo("Tuple", self.typing_module)

    @cached_property
    def container(self) -> TypeInfo:
        return TypeInfo("Container", self.typing_module)

    @cached_property
    def sequence(self) -> TypeInfo:
        return TypeInfo("Sequence", self.typing_module)

    @cached_property
    def mutable_sequence(self) -> TypeInfo:
        return TypeInfo("MutableSequence", self.typing_module)

    @cached_property
    def dataclass_decorator(self) -> TypeInfo:
        return TypeInfo("dataclass", self.dataclasses_module)

    @cached_property
    def typed_dict(self) -> TypeInfo:
        return TypeInfo("TypedDict", self.typing_module)

    @cached_property
    def mapping(self) -> TypeInfo:
        return TypeInfo("Mapping", self.typing_module)

    @cached_property
    def mutable_mapping(self) -> TypeInfo:
        return TypeInfo("MutableMapping", self.typing_module)

    @cached_property
    def optional(self) -> TypeInfo:
        return TypeInfo("Optional", self.typing_module)

    @cached_property
    def union(self) -> TypeInfo:
        return TypeInfo("Union", self.typing_module)

    @cached_property
    def context_manager(self) -> TypeInfo:
        return TypeInfo("ContextManager", self.typing_module)

    @cached_property
    def async_context_manager(self) -> TypeInfo:
        return TypeInfo("AsyncContextManager", self.typing_module)

    @cached_property
    def iterator(self) -> TypeInfo:
        return TypeInfo("Iterator", self.typing_module)

    @cached_property
    def async_iterator(self) -> TypeInfo:
        return TypeInfo("AsyncIterator", self.typing_module)

    @cached_property
    def iterable(self) -> TypeInfo:
        return TypeInfo("Iterable", self.typing_module)

    @cached_property
    def async_iterable(self) -> TypeInfo:
        return TypeInfo("AsyncIterable", self.typing_module)

    @cached_property
    def literal(self) -> TypeInfo:
        return TypeInfo("Literal", self.typing_module)

    @cached_property
    def no_return(self) -> TypeInfo:
        return TypeInfo("NoReturn", self.typing_module)

    @cached_property
    def overload_decorator(self) -> TypeInfo:
        return TypeInfo("overload", self.typing_module)

    @cached_property
    def override_decorator(self) -> TypeInfo:
        return TypeInfo("override", self.typing_module)
