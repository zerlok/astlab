from __future__ import annotations

__all__ = [
    "DefaultASTResolver",
]

import ast
import typing as t
from dataclasses import replace
from itertools import chain

from astlab._typing import assert_never, override
from astlab.abc import ASTExpressionBuilder, ASTResolver, ASTStatementBuilder, Stmt, TypeDefinitionBuilder, TypeExpr
from astlab.traverse import traverse_dfs_post_order
from astlab.types import (
    LiteralTypeInfo,
    ModuleInfo,
    NamedTypeInfo,
    TypeAnnotator,
    TypeInfo,
    TypeInspector,
    ellipsis_type_info,
    none_type_info,
)
from astlab.types.model import EnumTypeInfo, TypeVarInfo, UnionTypeInfo
from astlab.types.predef import get_predef
from astlab.version import PythonVersion


class DefaultASTResolver(ASTResolver):
    def __init__(
        self,
        inspector: t.Optional[TypeInspector] = None,
        annotator: t.Optional[TypeAnnotator] = None,
        python_version: t.Optional[PythonVersion] = None,
    ) -> None:
        self.__module: t.Optional[ModuleInfo] = None
        self.__namespace: t.Sequence[str] = ()
        self.__dependencies: t.MutableSet[ModuleInfo] = set[ModuleInfo]()
        self.__inspector = inspector if inspector is not None else TypeInspector()
        self.__annotator = annotator if annotator is not None else TypeAnnotator(python_version=python_version)
        self.__version = PythonVersion.get(python_version)

    @override
    def resolve_expr(self, expr: TypeExpr, *tail: str) -> ast.expr:
        if isinstance(expr, ast.expr):
            return self.__chain_attr(expr, *tail)

        elif isinstance(expr, ASTExpressionBuilder):
            return self.__chain_attr(expr.build_expr(), *tail)

        elif isinstance(expr, TypeDefinitionBuilder):
            return self.__resolve_info(expr.info, tail)

        else:
            info = (
                expr
                if isinstance(
                    expr, (ModuleInfo, TypeVarInfo, NamedTypeInfo, LiteralTypeInfo, EnumTypeInfo, UnionTypeInfo)
                )
                else self.__inspector.inspect(expr)
            )
            return self.__resolve_info(info, tail)

    @override
    def resolve_stmts(
        self,
        *stmts: t.Optional[Stmt],
        docs: t.Optional[t.Sequence[str]] = None,
        pass_if_empty: bool = False,
    ) -> list[ast.stmt]:
        body = list(
            chain.from_iterable(
                stmt.build_stmt()
                if isinstance(stmt, ASTStatementBuilder)
                else (stmt,)
                if isinstance(stmt, ast.stmt)
                else (ast.Expr(value=self.resolve_expr(stmt)),)
                for stmt in stmts
                if stmt is not None
            )
        )

        if docs:
            body.insert(0, ast.Expr(value=ast.Constant(value="\n".join(docs))))

        if not body and pass_if_empty:
            body.append(ast.Pass())

        return body

    @override
    def set_current_scope(
        self,
        module: t.Optional[ModuleInfo],
        namespace: t.Sequence[str],
        dependencies: t.MutableSet[ModuleInfo],
    ) -> None:
        self.__module = module
        self.__namespace = namespace
        self.__dependencies = dependencies

    def __resolve_info(self, root: TypeInfo, tail: t.Sequence[str] = ()) -> ast.expr:
        nodes = dict[TypeInfo, ast.expr]()

        for info in traverse_dfs_post_order(root, self.__get_children):
            resolved_info = self.__resolve_dependency(info)

            if isinstance(resolved_info, UnionTypeInfo):
                node = self.__build_union_expr(resolved_info, nodes)

            else:
                node = self.__build_expr(resolved_info, nodes)

            nodes[info] = node

        return self.__chain_attr(nodes[root], *tail)

    def __resolve_dependency(self, info: TypeInfo) -> TypeInfo:
        if isinstance(info, ModuleInfo):
            if info == self.__module:
                msg = "module cannot import itself"
                raise RuntimeError(msg, info)

            return info

        elif isinstance(info, (TypeVarInfo, NamedTypeInfo, EnumTypeInfo)):
            if info.module != self.__module:
                self.__dependencies.add(info.module)
                return info

            elif info.namespace[: len(self.__namespace)] == self.__namespace:
                # use shorten namespace for a type in nested namespace of the current scope
                return replace(info, namespace=info.namespace[len(self.__namespace) :])

            else:
                return info

        elif isinstance(info, LiteralTypeInfo):
            if info.module != self.__module:
                self.__dependencies.add(info.module)

            return info

        elif isinstance(info, UnionTypeInfo):
            return info

        else:
            assert_never(info)

    def __build_expr(self, info: TypeInfo, nodes: t.Mapping[TypeInfo, ast.expr]) -> ast.expr:
        if info == none_type_info():
            return ast.Constant(value=None)

        if info == ellipsis_type_info():
            return ast.Constant(value=...)

        parts = self.__module.parts if self.__module is not None else ()
        head, *tail = (
            (info.parts[len(parts) :] if info.module == self.__module else info.parts)
            if not isinstance(info, ModuleInfo)
            else info.parts
        )

        node = self.__chain_attr(ast.Name(id=head), *tail)

        if isinstance(info, NamedTypeInfo) and info.type_params:
            params = [nodes[tp] for tp in info.type_params]
            node = ast.Subscript(
                value=node,
                slice=ast.Tuple(elts=params) if len(params) > 1 else params[0],
            )

        if self.__is_forward_ref(info):
            node = ast.Constant(value=ast.unparse(node))

        return node

    def __build_union_expr(self, info: UnionTypeInfo, nodes: t.Mapping[TypeInfo, ast.expr]) -> ast.expr:
        if self.__version >= PythonVersion.PY314:
            head, *tail = info.values
            node = nodes[head]

            for val in tail:
                node = ast.BinOp(left=node, op=ast.BitOr(), right=nodes[val])

            return node

        else:
            union_named_type = get_predef().union.with_type_params(*info.values)
            return self.__build_expr(self.__resolve_dependency(union_named_type), nodes)

    def __is_forward_ref(self, info: TypeInfo) -> bool:
        return (
            self.__version < PythonVersion.PY314
            and not isinstance(info, ModuleInfo)
            and info.module == self.__module
            and (*info.namespace, info.name) == self.__namespace
        )

    def __chain_attr(self, expr: ast.expr, *tail: str) -> ast.expr:
        for attr in tail:
            expr = ast.Attribute(attr=attr, value=expr)

        return expr

    def __get_children(self, info: TypeInfo) -> t.Iterable[TypeInfo]:
        if isinstance(
            info,
            (ModuleInfo, TypeVarInfo, LiteralTypeInfo, EnumTypeInfo),
        ):
            return ()

        elif isinstance(info, NamedTypeInfo):
            return info.type_params

        elif isinstance(info, UnionTypeInfo):
            return info.values

        else:
            assert_never(info)
