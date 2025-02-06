from __future__ import annotations

__all__ = [
    "ASTExpressionBuilder",
    "ASTResolver",
    "ASTStatementBuilder",
    "Expr",
    "Stmt",
    "TypeDefBuilder",
    "TypeRef",
]


import abc
import ast
import typing as t

from astlab.info import RuntimeType, TypeInfo


class ASTExpressionBuilder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build(self) -> ast.expr:
        raise NotImplementedError


class ASTStatementBuilder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build(self) -> t.Sequence[ast.stmt]:
        raise NotImplementedError


class TypeDefBuilder(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def info(self) -> TypeInfo:
        raise NotImplementedError

    @abc.abstractmethod
    def ref(self) -> ASTExpressionBuilder:
        raise NotImplementedError


Expr = t.Union[ast.expr, ASTExpressionBuilder]
Stmt = t.Union[ast.stmt, ASTStatementBuilder, Expr]
TypeRef = t.Union[
    Expr,
    RuntimeType,
    TypeInfo,
    TypeDefBuilder,
]


class ASTResolver(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def expr(self, ref: TypeRef, *tail: str) -> ast.expr:
        raise NotImplementedError

    @abc.abstractmethod
    def body(
        self,
        *stmts: t.Optional[Stmt],
        docs: t.Optional[t.Sequence[str]] = None,
        pass_if_empty: bool = False,
    ) -> list[ast.stmt]:
        raise NotImplementedError
