from __future__ import annotations

__all__ = [
    "ModuleInfo",
    "PackageInfo",
    "RuntimeType",
    "TypeInfo",
]

import ast
import functools as ft
import importlib
import typing as t
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import GenericAlias

from astlab._typing import assert_never, override

if t.TYPE_CHECKING:
    from types import ModuleType

RuntimeType = t.Union[
    type[object],
    GenericAlias,
    t._SpecialForm,  # noqa: SLF001
    t._BaseGenericAlias,  # type: ignore[name-defined] # noqa: SLF001
]


@dataclass(frozen=True)
class PackageInfo:
    parent: t.Optional[PackageInfo]
    name: str

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

        info = cls(None, top)
        for name in tail:
            info = cls(info, name)

        return info

    @classmethod
    def build_or_none(cls, *parts: str) -> t.Optional[PackageInfo]:
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
    package: t.Optional[PackageInfo]
    name: str

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

        return cls(PackageInfo.build_or_none(*other), name)

    @classmethod
    def build_or_none(cls, *parts: str) -> t.Optional[ModuleInfo]:
        return cls.build(*parts) if parts else None

    @ft.cached_property
    def parts(self) -> t.Sequence[str]:
        return *(self.package.parts if self.package is not None else ()), self.name

    @ft.cached_property
    def qualname(self) -> str:
        return ".".join(self.parts)

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
    name: str
    module: ModuleInfo
    namespace: t.Sequence[str] = field(default_factory=tuple)
    type_params: t.Sequence[TypeInfo] = field(default_factory=tuple)
    type_vars: t.Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_str(cls, qualname: str) -> TypeInfo:
        if not qualname:
            msg = "qualname can't be empty"
            raise ValueError(msg)

        return TypeInfoParser().parse(qualname)

    @classmethod
    def from_type(cls, type_: RuntimeType) -> TypeInfo:
        return TypeInfoInspector().inspect(type_)

    @ft.cached_property
    def parts(self) -> t.Sequence[str]:
        return *self.module.parts, *self.namespace, self.name

    @ft.cached_property
    def qualname(self) -> str:
        return ".".join(self.parts)

    def with_type_params(self, *infos: TypeInfo) -> TypeInfo:
        if len(infos) > len(self.type_vars):
            msg = "too many type parameters"
            raise ValueError(msg, infos, self)

        return replace(
            self,
            type_params=(*self.type_params, *infos),
            type_vars=self.type_vars[len(infos) :],
        )


class TypeInfoParser(ast.NodeVisitor):
    def __init__(self) -> None:
        self.__module: t.Optional[ModuleInfo] = None
        self.__names = list[str]()
        self.__type_params = list[TypeInfo]()

    # NOTE: `ruff` can't work with `override`
    @override
    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if node.value is None:  # type: ignore[misc]
            self.__module = ModuleInfo(None, "builtins")
            self.__names.append("NoneType")

    # NOTE: `ruff` can't work with `override`
    @override
    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        info = self.__get_module_info(node.id)
        if info is not None:
            self.__module = info

        else:
            self.__module = ModuleInfo(None, "builtins")
            self.__names.append(node.id)

    # NOTE: `ruff` can't work with `override`
    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        self.generic_visit(node)

        info = self.__get_module_info(node.attr)
        if info is not None:
            self.__module = info

        elif info is None:
            self.__names.append(node.attr)

    # NOTE: `ruff` can't work with `override`
    @override
    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        self.visit(node.value)

        if isinstance(node.slice, ast.Tuple):
            for item in node.slice.elts:
                self.__parse_type_param(item)
        else:
            self.__parse_type_param(node.slice)

    def parse(self, qualname: str) -> TypeInfo:
        return self.build(ast.parse(qualname))

    def build(self, node: ast.AST) -> TypeInfo:
        self.visit(node)

        if not self.__names:
            msg = "invalid AST"
            raise ValueError(msg, ast.dump(node))

        *namespace, name = self.__names

        return TypeInfo(
            name=name,
            module=self.__module if self.__module else ModuleInfo(None, "builtins"),
            namespace=tuple(namespace),
            type_params=tuple(self.__type_params),
            type_vars=(),
        )

    def __get_module_info(self, name: str) -> t.Optional[ModuleInfo]:
        if self.__names:
            return None

        package = PackageInfo(self.__module.package, self.__module.name) if self.__module is not None else None
        info = ModuleInfo(package, name)

        try:
            importlib.import_module(info.qualname)

        except ImportError:
            return None

        else:
            return info

    def __parse_type_param(self, item: ast.AST) -> None:
        parser = self.__class__()
        type_param = parser.build(item)

        self.__type_params.append(type_param)


class TypeInfoInspector:
    def inspect(self, type_: RuntimeType) -> TypeInfo:
        if isinstance(
            type_,
            (
                type,  # type: ignore[misc]
                GenericAlias,  # type: ignore[misc]
            ),
        ):
            origin = self.__get_type_origin(type_)
            module = ModuleInfo.from_str(origin.__module__)
            *namespace, name = origin.__name__.split(".")

            return TypeInfo(
                name=name,
                module=module,
                namespace=tuple(namespace),
                type_params=tuple(self.inspect(param) for param in self.__get_type_params(type_)),
                type_vars=tuple(str(param) for param in self.__get_type_vars(type_)),
            )

        elif isinstance(
            type_,
            (
                t._SpecialForm,  # noqa: SLF001
                t._BaseGenericAlias,  # type: ignore[attr-defined,misc] # noqa: SLF001
            ),
        ):
            return TypeInfoParser().parse(str(type_))

        else:
            assert_never(type_)  # noqa: RET503

    def __get_type_origin(self, type_: RuntimeType) -> type[object]:
        typing_origin: t.Optional[type[object]] = t.get_origin(type_)
        origin = typing_origin or type_
        assert isinstance(origin, type)  # type: ignore[misc]
        return origin

    def __get_type_params(self, type_: RuntimeType) -> t.Sequence[RuntimeType]:
        origin: t.Optional[RuntimeType] = t.get_origin(type_)
        args: t.Optional[t.Sequence[RuntimeType]] = t.get_args(type_)

        # patch Union[T, None] => Optional[T]
        if origin is t.Union and args is not None and len(args) == 2 and args[1] is type(None):  # noqa: PLR2004
            return args[:1]

        return args or ()

    def __get_type_vars(self, type_: RuntimeType) -> t.Sequence[RuntimeType]:
        value: object = getattr(type_, "__parameters__", None)
        assert value is None or isinstance(value, tuple)
        return value or ()
