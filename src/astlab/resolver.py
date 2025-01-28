from __future__ import annotations

__all__ = [
    "DefaultASTResolver",
]

import ast
import typing as t
from itertools import chain

from astlab._typing import assert_never, override
from astlab.abc import ASTExpressionBuilder, ASTResolver, ASTStatementBuilder, Stmt, TypeDefBuilder, TypeRef
from astlab.info import ModuleInfo, TypeInfo

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
                t._SpecialForm,  # noqa: SLF001
            ),
        ):
            return self.__type_info_expr(self.__get_type_info(ref), *tail)

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
        resolved_info = self.__resolve_dependency(info)
        head, *middle = (*(resolved_info.module.parts if resolved_info.module is not None else ()), *resolved_info.ns)

        return self.__chain_attr(ast.Name(id=head), *middle, *tail)

    def __resolve_dependency(self, info: TypeInfo) -> TypeInfo:
        if info.module is not None and info.module != self.__context.module:
            self.__context.current_dependencies.add(info.module)
            return info

        ns = self.__context.namespace
        return TypeInfo(None, info.ns if info.ns[: len(ns)] != ns else info.ns[len(ns) :])

    # TODO: support type checking in this function, but keep the hack for typing.Optional in python 3.9
    @t.no_type_check
    def __get_type_info(self, obj: object) -> TypeInfo:
        if isinstance(obj, type):
            return TypeInfo.from_type(obj)

        if isinstance(
            obj,
            t._SpecialForm,  # noqa: SLF001
        ):
            return TypeInfo.build(
                ModuleInfo.from_str(obj.__module__),
                obj._name,  # noqa: SLF001
            )

        msg = "can't get type info for provided type"
        raise TypeError(msg, obj)

    def __chain_attr(self, expr: ast.expr, *tail: str) -> ast.expr:
        for attr in tail:
            expr = ast.Attribute(attr=attr, value=expr)

        return expr
