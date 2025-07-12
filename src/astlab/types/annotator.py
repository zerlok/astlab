from __future__ import annotations

__all__ = [
    "TypeAnnotator",
]

import ast
import typing as t
from collections import deque
from dataclasses import replace

from astlab._typing import assert_never, override
from astlab.cache import lru_cache_method
from astlab.types.loader import ModuleLoader
from astlab.types.model import (
    LiteralTypeInfo,
    LiteralTypeValue,
    ModuleInfo,
    NamedTypeInfo,
    TypeInfo,
    builtins_module_info,
    ellipsis_type_info,
    none_type_info,
)


class TypeAnnotator:
    """Provides annotation string form type info and vice versa (parses annotation to type info)."""

    def __init__(self, loader: t.Optional[ModuleLoader] = None) -> None:
        self.__loader = loader or ModuleLoader()

    @lru_cache_method()
    def annotate(self, info: TypeInfo) -> str:
        if isinstance(info, NamedTypeInfo):
            if info == none_type_info():
                return "None"

            if info == ellipsis_type_info():
                return "..."

            if not info.type_params:
                return info.qualname

            # TODO: fix recursive type
            params = ", ".join(self.annotate(tp) for tp in info.type_params)
            return f"{info.qualname}[{params}]"

        elif isinstance(info, LiteralTypeInfo):
            vals = ", ".join(repr(v) for v in info.values)
            return f"typing.Literal[{vals}]"

        else:
            assert_never(info)

    def parse(self, qualname: str) -> TypeInfo:
        node = ast.parse(qualname)

        if len(node.body) != 1:
            msg = "invalid qualified name"
            raise ValueError(msg, qualname)

        return _ExprParser(self.__loader).parse(node)


class _ExprParser(ast.NodeVisitor):
    def __init__(self, loader: ModuleLoader) -> None:
        self.__loader = loader
        self.__parts = deque[str]()
        self.__info: t.Optional[TypeInfo] = None

    # NOTE: `ruff` can't work with `override`
    @override
    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if node.value is None:  # type: ignore[misc]
            self.__set_result(none_type_info())

        elif node.value is Ellipsis:  # type: ignore[misc]
            self.__set_result(ellipsis_type_info())

        elif isinstance(node.value, str):  # type: ignore[misc]
            # forward ref case
            info = _ExprParser(self.__loader).parse(ast.parse(node.value))
            self.__set_result(info)

    # NOTE: `ruff` can't work with `override`
    @override
    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        self.__parts.appendleft(node.id)
        *parts, name = self.__parts

        module = self.__extract_module_info(node, parts)

        info = NamedTypeInfo(
            name=name,
            module=module,
            namespace=tuple(parts[len(module.parts) :]),
        )

        self.__set_result(info)

    # NOTE: `ruff` can't work with `override`
    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        self.__parts.appendleft(node.attr)
        self.visit(node.value)

    # NOTE: `ruff` can't work with `override`
    @override
    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        self.visit(node.value)

        if isinstance(self.__info, NamedTypeInfo):
            if self.__info.qualname == "typing.Literal":
                self.__info = LiteralTypeInfo(values=tuple(_LiteralValueExtractor().extract(node)))

            else:
                self.__info = replace(
                    self.__info,
                    type_params=tuple(
                        self.__parse_type_params(*node.slice.elts)
                        if isinstance(node.slice, ast.Tuple)
                        else self.__parse_type_params(node.slice)
                    ),
                    type_vars=(),
                )

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

    def __extract_module_info(self, node: ast.AST, parts: t.Sequence[str]) -> ModuleInfo:
        if not parts:
            return builtins_module_info()

        for i in range(len(parts), 0, -1):
            module = ModuleInfo.build(*parts[:i])
            try:
                self.__loader.load(module)
            except ImportError:
                continue
            else:
                return module

        msg = "invalid module parts"
        raise ValueError(msg, parts, ast.dump(node))

    def __parse_type_params(self, *items: ast.AST) -> t.Sequence[TypeInfo]:
        return [self.__class__(self.__loader).parse(item) for item in items]


class _LiteralValueExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.__values = list[LiteralTypeValue]()

    # NOTE: `ruff` can't work with `override`
    @override
    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if node.value is not None and not isinstance(node.value, (bool, int, bytes, str)):  # type: ignore[misc]
            msg = "invalid literal value"
            raise ValueError(msg, ast.dump(node))

        self.__values.append(node.value)

    def extract(self, node: ast.AST) -> t.Sequence[LiteralTypeValue]:
        self.visit(node)
        return self.__values
