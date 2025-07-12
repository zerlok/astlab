from __future__ import annotations

__all__ = [
    "DefaultASTResolver",
]

import ast
import typing as t
from dataclasses import replace
from itertools import chain

from astlab._typing import assert_never, override
from astlab.abc import ASTExpressionBuilder, ASTResolver, ASTStatementBuilder, Stmt, TypeDefinitionBuilder, TypeRef
from astlab.types import LiteralTypeInfo, ModuleInfo, NamedTypeInfo, TypeInfo, TypeInspector


class DefaultASTResolver(ASTResolver):
    def __init__(self, inspector: t.Optional[TypeInspector] = None) -> None:
        self.__module: t.Optional[ModuleInfo] = None
        self.__namespace: t.Sequence[str] = ()
        self.__dependencies: t.MutableSet[ModuleInfo] = set[ModuleInfo]()
        self.__inspector = inspector if inspector is not None else TypeInspector()

    @override
    def resolve_expr(self, ref: TypeRef, *tail: str) -> ast.expr:
        if isinstance(ref, ast.expr):
            return self.__chain_attr(ref, *tail)

        elif isinstance(ref, ASTExpressionBuilder):
            return self.__chain_attr(ref.build_expr(), *tail)

        elif isinstance(ref, (NamedTypeInfo, LiteralTypeInfo)):
            return self.__type_info_expr(ref, *tail)

        elif isinstance(ref, TypeDefinitionBuilder):
            return self.__type_info_expr(ref.info, *tail)

        else:
            info = self.__inspector.inspect(ref)
            return self.__type_info_expr(info, *tail)

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

    def __type_info_expr(self, info: TypeInfo, *tail: str) -> ast.expr:
        if isinstance(info, NamedTypeInfo) and info.type_vars:
            msg = "can't build expr for type with type vars"
            raise ValueError(msg, info)

        resolved_info = self.__resolve_dependency(info)
        return self.__type_info_attr(resolved_info, *tail)

    def __type_info_attr(self, info: TypeInfo, *tail: str) -> ast.expr:
        parts = self.__module.parts if self.__module is not None else ()
        head, *middle = info.parts[len(parts) :] if info.module == self.__module else info.parts

        origin = self.__chain_attr(ast.Name(id=head), *middle, *tail)
        args = (
            [self.__type_info_attr(param) for param in info.type_params]
            if isinstance(info, NamedTypeInfo)
            else [ast.Constant(value=value) for value in info.values]
        )

        return (
            ast.Subscript(
                value=origin,
                slice=ast.Tuple(elts=args) if len(args) > 1 else args[0],
            )
            if args
            else origin
        )

    def __resolve_dependency(self, info: TypeInfo) -> TypeInfo:
        if isinstance(info, NamedTypeInfo):
            if info.module == self.__module:
                ns = (
                    info.namespace[len(self.__namespace) :]
                    if info.namespace[: len(self.__namespace)] == self.__namespace
                    else info.namespace
                )

            else:
                self.__dependencies.add(info.module)
                ns = info.namespace

            return replace(
                info,
                namespace=ns,
                type_params=tuple(self.__resolve_dependency(param) for param in info.type_params),
            )

        elif isinstance(info, LiteralTypeInfo):
            self.__dependencies.add(info.module)
            return info

        else:
            assert_never(info)

    def __chain_attr(self, expr: ast.expr, *tail: str) -> ast.expr:
        for attr in tail:
            expr = ast.Attribute(attr=attr, value=expr)

        return expr
