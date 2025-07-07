from __future__ import annotations

__all__ = [
    "ASTExpressionBuilder",
    "ASTLabError",
    "ASTResolver",
    "ASTStatementBuilder",
    "Expr",
    "Stmt",
    "TypeDefinitionBuilder",
    "TypeRef",
]


import abc
import ast
import typing as t

from astlab.types import ModuleInfo, RuntimeType, TypeInfo

if t.TYPE_CHECKING:
    from astlab._typing import TypeAlias


class ASTLabError(Exception):
    """Base exception for all errors in AST lab library."""


class ASTExpressionBuilder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build_expr(self) -> ast.expr:
        raise NotImplementedError


class ASTStatementBuilder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        raise NotImplementedError


class TypeDefinitionBuilder(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def info(self) -> TypeInfo:
        raise NotImplementedError

    @abc.abstractmethod
    def ref(self) -> ASTExpressionBuilder:
        raise NotImplementedError


Expr: TypeAlias = t.Union[ast.expr, ASTExpressionBuilder]
Stmt: TypeAlias = t.Union[ast.stmt, ASTStatementBuilder, Expr]
TypeRef: TypeAlias = t.Union[
    Expr,
    RuntimeType,
    TypeInfo,
    TypeDefinitionBuilder,
]


class ASTResolver(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def resolve_expr(self, ref: TypeRef, *tail: str) -> ast.expr:
        raise NotImplementedError

    @abc.abstractmethod
    def resolve_stmts(
        self,
        *stmts: t.Optional[Stmt],
        docs: t.Optional[t.Sequence[str]] = None,
        pass_if_empty: bool = False,
    ) -> list[ast.stmt]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_current_scope(
        self,
        module: t.Optional[ModuleInfo],
        namespace: t.Sequence[str],
        dependencies: t.MutableSet[ModuleInfo],
    ) -> None:
        raise NotImplementedError
