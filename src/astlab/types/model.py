from __future__ import annotations

__all__ = [
    "LiteralTypeInfo",
    "LiteralTypeValue",
    "ModuleInfo",
    "NamedTypeInfo",
    "PackageInfo",
    "RuntimeType",
    "TypeInfo",
    "builtins_module_info",
    "ellipsis_type_info",
    "none_type_info",
    "typing_module_info",
]

import typing as t
from dataclasses import dataclass, field, replace
from functools import cache, cached_property
from pathlib import Path
from types import GenericAlias, ModuleType

if t.TYPE_CHECKING:
    from astlab._typing import TypeAlias

RuntimeType: TypeAlias = t.Union[
    type[object],
    GenericAlias,
    t._SpecialForm,  # noqa: SLF001
    t._BaseGenericAlias,  # type: ignore[name-defined] # noqa: SLF001
    t._LiteralGenericAlias,  # type: ignore[name-defined] # noqa: SLF001
]


@dataclass(frozen=True)
class PackageInfo:
    name: str
    parent: t.Optional[PackageInfo] = None

    @classmethod
    def from_str(cls, qualname: str) -> PackageInfo:
        if not qualname:
            msg = "qualname can't be empty"
            raise ValueError(msg)

        return cls.build(*qualname.split("."))

    @classmethod
    def from_module(cls, obj: ModuleType) -> PackageInfo:
        return cls.from_str(
            obj.__name__
            if obj.__file__ is not None and obj.__file__.endswith("__init__.py")
            else obj.__name__.rpartition(".")[0]
        )

    @classmethod
    def build(cls, *parts: str) -> PackageInfo:
        if not parts:
            msg = "at least one part should be provided for package name"
            raise ValueError(msg)

        top, *tail = parts

        info = cls(top)
        for name in tail:
            info = cls(name, info)

        return info

    @classmethod
    def build_or_none(cls, *parts: str) -> t.Optional[PackageInfo]:
        return cls.build(*parts) if parts else None

    @cached_property
    def parts(self) -> t.Sequence[str]:
        return *(self.parent.parts if self.parent is not None else ()), self.name

    @cached_property
    def qualname(self) -> str:
        return ".".join(self.parts)

    @cached_property
    def directory(self) -> Path:
        return Path(*self.parts)


@dataclass(frozen=True)
class ModuleInfo:
    name: str
    package: t.Optional[PackageInfo] = None

    @classmethod
    def from_str(cls, qualname: str) -> ModuleInfo:
        if not qualname:
            msg = "qualname can't be empty"
            raise ValueError(msg)

        return cls.build(*qualname.split("."))

    @classmethod
    def from_module(cls, obj: ModuleType) -> ModuleInfo:
        return cls.from_str(obj.__name__)

    @classmethod
    def build(cls, *parts: str) -> ModuleInfo:
        if not parts:
            msg = "at least one part should be provided for module name"
            raise ValueError(msg)

        *other, name = parts

        return cls(name, PackageInfo.build_or_none(*other))

    @classmethod
    def build_or_none(cls, *parts: str) -> t.Optional[ModuleInfo]:
        return cls.build(*parts) if parts else None

    @cached_property
    def parts(self) -> t.Sequence[str]:
        return *(self.package.parts if self.package is not None else ()), self.name

    @cached_property
    def qualname(self) -> str:
        return ".".join(self.parts)

    @cached_property
    def file(self) -> Path:
        return ((self.package.directory / self.name) if self.package is not None else Path(self.name)).with_suffix(
            ".py",
        )

    @cached_property
    def stub_file(self) -> Path:
        return self.file.with_suffix(".pyi")


@cache  # type: ignore[misc]
def builtins_module_info() -> ModuleInfo:
    return ModuleInfo("builtins")


@cache  # type: ignore[misc]
def typing_module_info() -> ModuleInfo:
    return ModuleInfo("typing")


@dataclass(frozen=True)
class NamedTypeInfo:
    name: str
    module: ModuleInfo
    namespace: t.Sequence[str] = field(default_factory=tuple)
    type_params: t.Sequence[TypeInfo] = field(default_factory=tuple)
    type_vars: t.Sequence[str] = field(default_factory=tuple)

    @classmethod
    def build(
        cls,
        module: t.Union[str, t.Sequence[str], ModuleInfo],
        name: str,
        type_params: t.Optional[t.Sequence[TypeInfo]] = None,
        type_vars: t.Optional[t.Sequence[str]] = None,
    ) -> NamedTypeInfo:
        *namespace, type_name = name.split(".")
        if not type_name:
            msg = "type name can't be empty"
            raise ValueError(msg, name)

        return NamedTypeInfo(
            name=type_name,
            module=module
            if isinstance(module, ModuleInfo)
            else ModuleInfo.from_str(module)
            if isinstance(module, str)
            else ModuleInfo.build(*module),
            namespace=tuple(namespace),
            type_params=tuple(type_params or ()),
            type_vars=tuple(type_vars or ()),
        )

    @cached_property
    def parts(self) -> t.Sequence[str]:
        return *self.module.parts, *self.namespace, self.name

    @cached_property
    def qualname(self) -> str:
        return ".".join(self.parts)

    def with_type_params(self, *infos: TypeInfo) -> NamedTypeInfo:
        if not infos:
            return self

        if len(infos) > len(self.type_vars):
            msg = "too many type parameters"
            raise ValueError(msg, infos, self)

        return replace(
            self,
            type_params=(*self.type_params, *infos),
            type_vars=self.type_vars[len(infos) :],
        )


@cache  # type: ignore[misc]
def none_type_info() -> NamedTypeInfo:
    return NamedTypeInfo("NoneType", builtins_module_info())


@cache  # type: ignore[misc]
def ellipsis_type_info() -> NamedTypeInfo:
    return NamedTypeInfo("ellipsis", builtins_module_info())


LiteralTypeValue: TypeAlias = t.Union[bool, int, bytes, str, None]


@dataclass(frozen=True)
class LiteralTypeInfo:
    # TODO: enum values
    values: t.Sequence[LiteralTypeValue]

    @cached_property
    def name(self) -> str:
        return "Literal"

    @cached_property
    def module(self) -> ModuleInfo:
        return typing_module_info()

    @cached_property
    def namespace(self) -> t.Sequence[str]:
        return ()

    @cached_property
    def parts(self) -> t.Sequence[str]:
        return *self.module.parts, *self.namespace, self.name

    @cached_property
    def qualname(self) -> str:
        return ".".join(self.parts)


TypeInfo: TypeAlias = t.Union[NamedTypeInfo, LiteralTypeInfo]
