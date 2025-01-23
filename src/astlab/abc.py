from __future__ import annotations

__all__ = [
    "ExpressionASTBuilder",
    "StatementASTBuilder",
    "TypeInfoProvider",
]


import abc
import typing as t

if t.TYPE_CHECKING:
    import ast

    from astlab.info import TypeInfo


class ExpressionASTBuilder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build(self) -> ast.expr:
        raise NotImplementedError


class StatementASTBuilder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build(self) -> t.Sequence[ast.stmt]:
        raise NotImplementedError


class TypeInfoProvider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def provide_type_info(self) -> TypeInfo:
        raise NotImplementedError
