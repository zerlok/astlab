from __future__ import annotations

__all__ = [
    "TypeAnnotator",
]

import ast
import enum
import typing as t
from collections import deque
from dataclasses import replace

from astlab._typing import assert_never, override
from astlab.cache import lru_cache_method
from astlab.types.loader import TypeLoader, TypeLoaderError
from astlab.types.model import (
    EnumTypeInfo,
    EnumTypeValue,
    LiteralTypeInfo,
    LiteralTypeValue,
    ModuleInfo,
    NamedTypeInfo,
    RuntimeType,
    TypeInfo,
    TypeVarInfo,
    UnionTypeInfo,
    builtins_module_info,
    ellipsis_type_info,
    none_type_info,
)
from astlab.types.predef import get_predef
from astlab.version import PythonVersion


# TODO: consider ast.unparse usage for building valid type annotation
class TypeAnnotator:
    """Provides annotation string form type info and vice versa (parses annotation to type info)."""

    def __init__(
        self,
        loader: t.Optional[TypeLoader] = None,
        python_version: t.Optional[PythonVersion] = None,
    ) -> None:
        self.__loader = loader or TypeLoader(python_version=python_version)
        self.__version = PythonVersion.get(python_version)

    @lru_cache_method()
    def annotate(self, info: TypeInfo) -> str:
        if info == none_type_info():
            return "None"

        if info == ellipsis_type_info():
            return "..."

        if isinstance(info, ModuleInfo):
            annotation = "builtins.module"

        elif isinstance(info, TypeVarInfo):
            annotation = info.qualname

        elif isinstance(info, NamedTypeInfo):
            annotation = info.qualname

            if info.type_params:
                # TODO: fix recursive type
                params = ", ".join(self.annotate(tp) for tp in info.type_params)
                annotation = f"{annotation}[{params}]"

        elif isinstance(info, LiteralTypeInfo):
            vals = ", ".join(repr(v) for v in info.values)
            annotation = f"{info.qualname}[{vals}]"

        elif isinstance(info, EnumTypeInfo):
            annotation = info.qualname

        elif isinstance(info, UnionTypeInfo):
            annotation = self.__annotate_union_type(info)

        else:
            assert_never(info)

        return annotation

    @lru_cache_method()
    def parse(self, qualname: str) -> TypeInfo:
        node = ast.parse(qualname)

        if len(node.body) != 1:
            msg = "invalid qualified name"
            raise ValueError(msg, qualname)

        return _ExprParser(self.__loader).parse(node)

    def __annotate_union_type(self, info: UnionTypeInfo) -> str:
        if self.__version >= PythonVersion.PY310:
            return " | ".join(self.annotate(val) for val in info.values)

        args = ", ".join(self.annotate(val) for val in info.values)
        return f"{get_predef().union.qualname}[{args}]" if args else get_predef().union.qualname


class _ExprParser(ast.NodeVisitor):
    def __init__(self, loader: TypeLoader) -> None:
        self.__loader = loader
        self.__parts = deque[str]()
        self.__info: t.Optional[TypeInfo] = None

    @override
    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value is None:
            self.__set_result(none_type_info())

        elif node.value is Ellipsis:
            self.__set_result(ellipsis_type_info())

        elif isinstance(node.value, str):
            # forward ref case
            info = _ExprParser(self.__loader).parse(ast.parse(node.value))
            self.__set_result(info)

    @override
    def visit_Name(self, node: ast.Name) -> None:
        self.__parts.appendleft(node.id)
        *parts, name = self.__parts
        named_type_info = self.__extract_named_type_info(node, parts, name)
        rtt: RuntimeType = self.__loader.load(named_type_info)

        if isinstance(rtt, t.TypeVar):  # type: ignore[misc]
            self.__set_result(
                TypeVarInfo(
                    name=named_type_info.name,
                    module=named_type_info.module,
                    namespace=named_type_info.namespace,
                )
            )

        elif isinstance(rtt, type) and issubclass(rtt, enum.Enum):  # type: ignore[misc]
            self.__set_result(
                EnumTypeInfo(
                    name=named_type_info.name,
                    module=named_type_info.module,
                    namespace=named_type_info.namespace,
                    values=tuple(
                        EnumTypeValue(
                            name=enum_value.name,
                            value=enum_value.value,  # type: ignore[misc]
                        )
                        for enum_value in rtt
                    ),
                )
            )

        else:
            self.__set_result(named_type_info)

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.__parts.appendleft(node.attr)
        self.visit(node.value)

    @override
    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.visit(node.value)

        if isinstance(self.__info, NamedTypeInfo):
            if self.__info.qualname == "typing.Literal":
                self.__info = LiteralTypeInfo(values=tuple(_LiteralValueExtractor().extract(node)))

            else:
                self.__info = replace(
                    self.__info,
                    type_params=tuple(
                        self.__parse_nested(*node.slice.elts)
                        if isinstance(node.slice, ast.Tuple)
                        else self.__parse_nested(node.slice)
                    ),
                )

    @override
    def visit_BinOp(self, node: ast.BinOp) -> None:
        left, right = self.__parse_nested(node.left, node.right)

        values = list[TypeInfo]()
        values.extend(left.values if isinstance(left, UnionTypeInfo) else [left])
        values.extend(right.values if isinstance(right, UnionTypeInfo) else [right])

        self.__set_result(UnionTypeInfo(values=tuple(values)))

    def parse(self, node: ast.AST) -> TypeInfo:
        self.visit(node)

        if self.__info is None:
            msg = "can't parse type info"
            raise ValueError(msg, ast.dump(node))

        return self.__info

    def __set_result(self, info: TypeInfo) -> None:
        if self.__info is not None:
            msg = "result is set already"
            raise RuntimeError(msg, self.__info, info)

        self.__info = info

    def __extract_named_type_info(
        self,
        node: ast.AST,
        parts: t.Sequence[str],
        name: str,
    ) -> NamedTypeInfo:
        if not parts:
            return NamedTypeInfo(
                name=name,
                module=builtins_module_info(),
            )

        for i in range(len(parts), 0, -1):
            module = ModuleInfo.build(*parts[:i])
            try:
                self.__loader.load(module)
            except TypeLoaderError:
                continue
            else:
                return NamedTypeInfo(
                    name=name,
                    module=module,
                    namespace=tuple(parts[len(module.parts) :]),
                )

        msg = "invalid module parts"
        raise ValueError(msg, parts, ast.dump(node))

    def __parse_nested(self, *items: ast.AST) -> t.Sequence[TypeInfo]:
        return [self.__class__(self.__loader).parse(item) for item in items]


class _LiteralValueExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.__values = list[LiteralTypeValue]()

    @override
    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value is not None and not isinstance(node.value, (bool, int, bytes, str)):
            msg = "invalid literal value"
            raise ValueError(msg, ast.dump(node))

        self.__values.append(node.value)

    def extract(self, node: ast.AST) -> t.Sequence[LiteralTypeValue]:
        self.visit(node)
        return self.__values
