from __future__ import annotations

import ast
import typing as t
from itertools import chain

from astlab.abc import ExpressionASTBuilder, StatementASTBuilder, TypeInfoProvider
from astlab.info import TypeInfo

if t.TYPE_CHECKING:
    from astlab.context import BuildContext
    from astlab.types import Stmt, TypeRef


class Resolver:
    def __init__(self, context: BuildContext) -> None:
        self.__context = context

    def expr(self, ref: TypeRef, *tail: str) -> ast.expr:
        if isinstance(ref, ExpressionASTBuilder):
            ref = ref.build()

        if isinstance(ref, ast.expr):
            return self.__chain_attr(ref, *tail)

        if isinstance(ref, TypeInfoProvider):
            ref = ref.provide_type_info()

        if not isinstance(ref, TypeInfo):
            ref = TypeInfo.from_type(ref)

        assert isinstance(ref, TypeInfo), f"{type(ref)}: {ref}"

        if ref.module is not None and ref.module != self.__context.module:
            self.__context.current_dependencies.add(ref.module)

        else:
            ns = self.__context.namespace
            ref = TypeInfo(None, ref.ns if ref.ns[: len(ns)] != ns else ref.ns[len(ns) :])

        head, *middle = (*(ref.module.parts if ref.module is not None else ()), *ref.ns)

        return self.__chain_attr(ast.Name(id=head), *middle, *tail)

    def stmts(
        self,
        *stmts: t.Optional[Stmt],
        docs: t.Optional[t.Sequence[str]] = None,
        pass_if_empty: bool = False,
    ) -> list[ast.stmt]:
        body = list(
            chain.from_iterable(
                stmt.build()
                if isinstance(stmt, StatementASTBuilder)
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

    def __chain_attr(self, expr: ast.expr, *tail: str) -> ast.expr:
        for attr in tail:
            expr = ast.Attribute(attr=attr, value=expr)

        return expr
