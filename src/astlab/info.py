from __future__ import annotations

__all__ = [
    "ModuleInfo",
    "PackageInfo",
    "TypeInfo",
]

import functools as ft
import typing as t
from dataclasses import dataclass
from pathlib import Path

if t.TYPE_CHECKING:
    from types import ModuleType

    from astlab._typing import Self


@dataclass(frozen=True)
class PackageInfo:
    parent: t.Optional[PackageInfo]
    name: str

    @classmethod
    def build(cls, *parts: str) -> Self:
        top, *tail = parts

        info = cls(None, top)
        for name in tail:
            info = cls(info, name)

        return info

    @classmethod
    def build_or_none(cls, *parts: str) -> t.Optional[Self]:
        return cls.build(*parts) if parts else None

    @ft.cached_property
    def parts(self) -> t.Sequence[str]:
        return *(self.parent.parts if self.parent is not None else ()), self.name

    @ft.cached_property
    def qualname(self) -> str:
        return ".".join(self.parts)

    @ft.cached_property
    def directory(self) -> Path:
        return Path(*self.parts)


@dataclass(frozen=True)
class ModuleInfo:
    parent: t.Optional[PackageInfo]
    name: str

    @classmethod
    def from_str(cls, ref: str) -> Self:
        *other, last = ref.split(".")
        return cls(PackageInfo.build_or_none(*other), last)

    @classmethod
    def from_module(cls, obj: ModuleType) -> Self:
        return cls.from_str(obj.__name__)

    @ft.cached_property
    def parts(self) -> t.Sequence[str]:
        return *(self.parent.parts if self.parent is not None else ()), self.name

    @ft.cached_property
    def qualname(self) -> str:
        return ".".join(self.parts)

    @property
    def package(self) -> t.Optional[PackageInfo]:
        return self.parent

    @ft.cached_property
    def file(self) -> Path:
        return ((self.package.directory / self.name) if self.package is not None else Path(self.name)).with_suffix(
            ".py",
        )

    @ft.cached_property
    def stub_file(self) -> Path:
        return self.file.with_suffix(".pyi")


@dataclass(frozen=True)
class TypeInfo:
    module: t.Optional[ModuleInfo]
    ns: t.Sequence[str]

    @classmethod
    def from_str(cls, ref: str) -> Self:
        module, ns = ref.split(":", maxsplit=1)
        return cls(ModuleInfo.from_str(module), tuple(ns.split(".")))

    @classmethod
    def from_type(cls, type_: type[object]) -> Self:
        return cls(ModuleInfo.from_str(type_.__module__), tuple(type_.__qualname__.split(".")))

    @classmethod
    def build(cls, module: t.Optional[ModuleInfo], *ns: str) -> Self:
        return cls(module, ns)
