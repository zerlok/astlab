from __future__ import annotations

__all__ = [
    "ModuleInfo",
    "PackageInfo",
    "TypeInfo",
]

import ast
import functools as ft
import importlib
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from astlab._typing import Self, override

if t.TYPE_CHECKING:
    from types import ModuleType


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
    type_params: t.Sequence[TypeInfo] = field(default_factory=tuple)

    @classmethod
    def from_str(cls, ref: str) -> Self:
        parser = TypeInfoParser()
        parser.visit(ast.parse(ref))

        return cls(
            module=parser.module,
            ns=parser.namespace,
            type_params=parser.type_params,
        )

    @classmethod
    def from_type(cls, type_: type[object]) -> Self:
        return cls(ModuleInfo.from_str(type_.__module__), tuple(type_.__qualname__.split(".")))

    @classmethod
    def build(cls, module: t.Optional[ModuleInfo], *ns: str) -> Self:
        return cls(module, ns)


class TypeInfoParser(ast.NodeVisitor):
    def __init__(self) -> None:
        self.__module: t.Optional[ModuleInfo] = None
        self.__namespace = list[str]()
        self.__type_params = list[TypeInfo]()

    @property
    def module(self) -> t.Optional[ModuleInfo]:
        return self.__module

    @property
    def namespace(self) -> t.Sequence[str]:
        return tuple(self.__namespace)

    @property
    def type_params(self) -> t.Sequence[TypeInfo]:
        return tuple(self.__type_params)

    # NOTE: `ruff` can't work with `override`
    @override
    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        info = self.__get_module_info(node.id)
        if info is not None:
            self.__module = info

    # NOTE: `ruff` can't work with `override`
    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        self.generic_visit(node)

        info = self.__get_module_info(node.attr)
        if info is not None:
            self.__module = info

        elif info is None:
            self.__namespace.append(node.attr)

    # NOTE: `ruff` can't work with `override`
    @override
    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        self.visit(node.value)

        if isinstance(node.slice, ast.Tuple):
            for item in node.slice.elts:
                self.__parse_type_param(item)
        else:
            self.__parse_type_param(node.slice)

    def __get_module_info(self, name: str) -> t.Optional[ModuleInfo]:
        if self.__namespace:
            return None

        package = PackageInfo(self.__module.parent, self.__module.name) if self.__module is not None else None
        info = ModuleInfo(package, name)

        try:
            importlib.import_module(info.qualname)

        except ImportError:
            return None

        else:
            return info

    def __parse_type_param(self, item: ast.AST) -> None:
        parser = self.__class__()
        parser.visit(item)
        self.__type_params.append(
            TypeInfo(
                module=parser.module,
                ns=parser.namespace,
                type_params=parser.type_params,
            )
        )
