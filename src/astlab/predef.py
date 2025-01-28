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
        return TypeInfo.build(self.contextlib_module, "asynccontextmanager")

    @cached_property
    def context_manager_decorator(self) -> TypeInfo:
        return TypeInfo.build(self.contextlib_module, "contextmanager")

    @cached_property
    def none_type(self) -> TypeInfo:
        return TypeInfo.build(self.builtins_module, "NoneType")

    @cached_property
    def bool(self) -> TypeInfo:
        return TypeInfo.build(self.builtins_module, "bool")

    @cached_property
    def int(self) -> TypeInfo:
        return TypeInfo.build(self.builtins_module, "int")

    @cached_property
    def float(self) -> TypeInfo:
        return TypeInfo.build(self.builtins_module, "float")

    @cached_property
    def str(self) -> TypeInfo:
        return TypeInfo.build(self.builtins_module, "str")

    @cached_property
    def list(self) -> TypeInfo:
        return TypeInfo.build(self.builtins_module, "list")

    @cached_property
    def dict(self) -> TypeInfo:
        return TypeInfo.build(self.builtins_module, "dict")

    @cached_property
    def set(self) -> TypeInfo:
        return TypeInfo.build(self.builtins_module, "set")

    @cached_property
    def property(self) -> TypeInfo:
        return TypeInfo.build(self.builtins_module, "property")

    @cached_property
    def classmethod(self) -> TypeInfo:
        return TypeInfo.build(self.builtins_module, "classmethod")

    @cached_property
    def abc_meta(self) -> TypeInfo:
        return TypeInfo.build(self.abc_module, "ABCMeta")

    @cached_property
    def abstractmethod(self) -> TypeInfo:
        return TypeInfo.build(self.abc_module, "abstractmethod")

    @cached_property
    def generic(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "Generic")

    @cached_property
    def final_decorator(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "final")

    @cached_property
    def final(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "Final")

    @cached_property
    def class_var(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "ClassVar")

    @cached_property
    def type(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "Type")

    @cached_property
    def tuple(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "Tuple")

    @cached_property
    def container(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "Container")

    @cached_property
    def sequence(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "Sequence")

    @cached_property
    def mutable_sequence(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "MutableSequence")

    @cached_property
    def dataclass_decorator(self) -> TypeInfo:
        return TypeInfo.build(self.dataclasses_module, "dataclass")

    @cached_property
    def typed_dict(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "TypedDict")

    @cached_property
    def mapping(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "Mapping")

    @cached_property
    def mutable_mapping(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "MutableMapping")

    @cached_property
    def optional(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "Optional")

    @cached_property
    def union(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "Union")

    @cached_property
    def context_manager(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "ContextManager")

    @cached_property
    def async_context_manager(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "AsyncContextManager")

    @cached_property
    def iterator(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "Iterator")

    @cached_property
    def async_iterator(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "AsyncIterator")

    @cached_property
    def iterable(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "Iterable")

    @cached_property
    def async_iterable(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "AsyncIterable")

    @cached_property
    def literal(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "Literal")

    @cached_property
    def no_return(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "NoReturn")

    @cached_property
    def overload_decorator(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "overload")

    @cached_property
    def override_decorator(self) -> TypeInfo:
        return TypeInfo.build(self.typing_module, "override")
