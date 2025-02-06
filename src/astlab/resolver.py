from __future__ import annotations

__all__ = [
    "DefaultASTResolver",
]

import ast
import typing as t
from dataclasses import replace
from itertools import chain
from types import GenericAlias

from astlab._typing import assert_never, override
from astlab.abc import ASTExpressionBuilder, ASTResolver, ASTStatementBuilder, Stmt, TypeDefBuilder, TypeRef
from astlab.info import TypeInfo

if t.TYPE_CHECKING:
    from astlab.context import BuildContext


class DefaultASTResolver(ASTResolver):
    def __init__(self, context: BuildContext) -> None:
        self.__context = context

    @override
    def expr(self, ref: TypeRef, *tail: str) -> ast.expr:
        if isinstance(ref, ast.expr):
            return self.__chain_attr(ref, *tail)

        elif isinstance(ref, ASTExpressionBuilder):
            return self.__chain_attr(ref.build(), *tail)

        # NOTE: type ignore fixes `Expression type contains "Any" (has type "type[type]")`
        elif isinstance(
            ref,
            (
                type,  # type: ignore[misc]
                GenericAlias,  # type: ignore[misc]
                t._SpecialForm,  # noqa: SLF001
                t._BaseGenericAlias,  # type: ignore[misc,attr-defined] # noqa: SLF001
            ),
        ):
            info = TypeInfo.from_type(ref)  # type: ignore[misc]
            return self.__type_info_expr(info, *tail)

        elif isinstance(ref, TypeInfo):
            return self.__type_info_expr(ref, *tail)

        elif isinstance(ref, TypeDefBuilder):
            return self.__type_info_expr(ref.info, *tail)

        else:
            # NOTE: it's `NoReturn` type
            assert_never(ref)  # noqa: RET503

    @override
    def body(
        self,
        *stmts: t.Optional[Stmt],
        docs: t.Optional[t.Sequence[str]] = None,
        pass_if_empty: bool = False,
    ) -> list[ast.stmt]:
        body = list(
            chain.from_iterable(
                stmt.build()
                if isinstance(stmt, ASTStatementBuilder)
                else (stmt,)
                if isinstance(stmt, ast.stmt)
                else (ast.Expr(value=self.expr(stmt)),)
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
        if info.type_vars:
            msg = "can't build expr for type with type vars"
            raise ValueError(msg, info)

        resolved_info = self.__resolve_dependency(info)
        return self.__type_info_attr(resolved_info, *tail)

    def __type_info_attr(self, info: TypeInfo, *tail: str) -> ast.expr:
        head, *middle = (
            info.parts[len(self.__context.module.parts) :] if info.module == self.__context.module else info.parts
        )

        origin = self.__chain_attr(ast.Name(id=head), *middle, *tail)
        args = [self.__type_info_attr(param) for param in info.type_params]

        return (
            ast.Subscript(
                value=origin,
                slice=ast.Tuple(elts=args) if len(args) > 1 else args[0],
            )
            if args
            else origin
        )

    def __resolve_dependency(self, info: TypeInfo) -> TypeInfo:
        type_params = tuple(self.__resolve_dependency(param) for param in info.type_params)

        if info.module != self.__context.module:
            self.__context.current_dependencies.add(info.module)
            return replace(info, type_params=type_params)

        ns = self.__context.namespace
        return replace(
            info,
            namespace=info.namespace[len(ns) :] if info.namespace[: len(ns)] == ns else info.namespace,
            type_params=type_params,
        )

    def __chain_attr(self, expr: ast.expr, *tail: str) -> ast.expr:
        for attr in tail:
            expr = ast.Attribute(attr=attr, value=expr)

        return expr
