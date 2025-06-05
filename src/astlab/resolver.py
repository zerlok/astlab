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
from astlab.types import LiteralTypeInfo, NamedTypeInfo, TypeInfo, TypeInspector

if t.TYPE_CHECKING:
    from astlab.context import BuildContext


class DefaultASTResolver(ASTResolver):
    def __init__(self, context: BuildContext, inspector: t.Optional[TypeInspector] = None) -> None:
        self.__context = context
        self.__inspector = inspector or TypeInspector()

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

    def __type_info_expr(self, info: TypeInfo, *tail: str) -> ast.expr:
        if isinstance(info, NamedTypeInfo) and info.type_vars:
            msg = "can't build expr for type with type vars"
            raise ValueError(msg, info)

        resolved_info = self.__resolve_dependency(info)
        return self.__type_info_attr(resolved_info, *tail)

    def __type_info_attr(self, info: TypeInfo, *tail: str) -> ast.expr:
        head, *middle = (
            info.parts[len(self.__context.module.parts) :] if info.module == self.__context.module else info.parts
        )

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
            if info.module == self.__context.module:
                ns = (
                    info.namespace[len(self.__context.namespace) :]
                    if info.namespace[: len(self.__context.namespace)] == self.__context.namespace
                    else info.namespace
                )

            else:
                self.__context.current_dependencies.add(info.module)
                ns = info.namespace

            return replace(
                info,
                namespace=ns,
                type_params=tuple(self.__resolve_dependency(param) for param in info.type_params),
            )

        elif isinstance(info, LiteralTypeInfo):
            self.__context.current_dependencies.add(info.module)
            return info

        else:
            assert_never(info)  # noqa: RET503

    def __chain_attr(self, expr: ast.expr, *tail: str) -> ast.expr:
        for attr in tail:
            expr = ast.Attribute(attr=attr, value=expr)

        return expr
