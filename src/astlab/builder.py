from __future__ import annotations

__all__ = [
    "AttrASTBuilder",
    "CallASTBuilder",
    "ClassScopeASTBuilder",
    "ClassStatementASTBuilder",
    "ClassTypeRefBuilder",
    "Comprehension",
    "ForStatementASTBuilder",
    "FuncStatementASTBuilder",
    "IfStatementASTBuilder",
    "MethodScopeASTBuilder",
    "MethodStatementASTBuilder",
    "ModuleASTBuilder",
    "PackageASTBuilder",
    "ScopeASTBuilder",
    "TryStatementASTBuilder",
    "WhileStatementASTBuilder",
    "WithStatementASTBuilder",
    "build_module",
    "build_package",
]

import abc
import ast
import sys
import typing as t
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property, partial, wraps
from itertools import chain

from typing_extensions import ParamSpec

from astlab._typing import Self, TypeGuard, override
from astlab.abc import (
    ASTExpressionBuilder,
    ASTLabError,
    ASTResolver,
    ASTStatementBuilder,
    Expr,
    Stmt,
    TypeDefinitionBuilder,
    TypeRef,
)
from astlab.context import BuildContext
from astlab.resolver import DefaultASTResolver
from astlab.types import (
    LiteralTypeInfo,
    ModuleInfo,
    NamedTypeInfo,
    PackageInfo,
    RuntimeType,
    TypeInfo,
    TypeInspector,
    predef,
)
from astlab.writer import render_module, write_module

T_co = t.TypeVar("T_co", covariant=True)
P = ParamSpec("P")


def build_package(
    info: t.Union[str, PackageInfo],
    parent: t.Optional[PackageInfo] = None,
    resolver: t.Optional[ASTResolver] = None,
    inspector: t.Optional[TypeInspector] = None,
) -> PackageASTBuilder:
    """Start python package builder."""
    return PackageASTBuilder(
        context=_create_context(resolver, inspector),
        info=info if isinstance(info, PackageInfo) else PackageInfo(info, parent),
        modules={},
    )


def build_module(
    info: t.Union[str, ModuleInfo],
    parent: t.Optional[PackageInfo] = None,
    resolver: t.Optional[ASTResolver] = None,
    inspector: t.Optional[TypeInspector] = None,
) -> ModuleASTBuilder:
    """Start python module builder."""
    return ModuleASTBuilder(
        context=_create_context(resolver, inspector),
        info=info if isinstance(info, ModuleInfo) else ModuleInfo(info, parent),
        body=[],
    )


def _create_context(
    resolver: t.Optional[ASTResolver],
    inspector: t.Optional[TypeInspector],
) -> BuildContext:
    inspector = inspector if inspector is not None else TypeInspector()

    return BuildContext(
        packages=[],
        dependencies=defaultdict(set),
        scopes=deque(),
        resolver=resolver if resolver is not None else DefaultASTResolver(inspector),
        inspector=inspector,
    )


class ASTBuildError(ASTLabError):
    pass


class IncompleteStatementError(ASTBuildError):
    pass


class _BaseBuilder:
    def __init__(self, context: BuildContext) -> None:
        self._context = context

    @property
    def _scope(self) -> ScopeASTBuilder:
        return ScopeASTBuilder(self._context)

    def _normalize_expr(self, expr: TypeRef, *tail: str) -> ast.expr:
        return self._context.resolver.resolve_expr(expr, *tail)

    def _normalize_body(self, body: t.Sequence[Stmt], docs: t.Optional[t.Sequence[str]] = None) -> list[ast.stmt]:
        return self._context.resolver.resolve_stmts(*body, docs=docs, pass_if_empty=True)


# noinspection PyTypeChecker
class BaseASTExpressionBuilder(_BaseBuilder, ASTExpressionBuilder):
    def __init__(self, context: BuildContext, factory: t.Callable[[], Expr]) -> None:
        super().__init__(context)
        self.__factory = factory

    def __neg__(self) -> Self:
        return self.__unary_op_expr(ast.Not())

    def __invert__(self) -> Self:
        return self.__unary_op_expr(ast.Invert())

    def __and__(self, other: Expr) -> Self:
        return self.__bool_op_expr(ast.And(), other)

    def __or__(self, other: Expr) -> Self:
        return self.__bool_op_expr(ast.Or(), other)

    def __add__(self, other: Expr) -> Self:
        return self.__bin_op_expr(ast.Add(), other)

    def __sub__(self, other: Expr) -> Self:
        return self.__bin_op_expr(ast.Sub(), other)

    def __mul__(self, other: Expr) -> Self:
        return self.__bin_op_expr(ast.Mult(), other)

    def __matmul__(self, other: Expr) -> Self:
        return self.__bin_op_expr(ast.MatMult(), other)

    def __truediv__(self, other: Expr) -> Self:
        return self.__bin_op_expr(ast.Div(), other)

    @override
    def build_expr(self) -> ast.expr:
        return self._normalize_expr(self.__factory())

    def stmt(self, *, append: bool = True) -> ast.stmt:
        node = ast.Expr(value=self.build_expr())
        if append:
            self._context.append_body(node)

        return node

    def __bool_op_expr(self, op: ast.boolop, right: Expr) -> Self:
        def create() -> ast.expr:
            return ast.BoolOp(op=op, values=[self.build_expr(), self._normalize_expr(right)])

        return self.__class__(context=self._context, factory=create)

    def __unary_op_expr(self, op: ast.unaryop) -> Self:
        def create() -> ast.expr:
            return ast.UnaryOp(op=op, operand=self.build_expr())

        return self.__class__(self._context, create)

    def __bin_op_expr(self, op: ast.operator, right: Expr) -> Self:
        def create() -> ast.expr:
            return ast.BinOp(left=self.build_expr(), op=op, right=self._normalize_expr(right))

        return self.__class__(self._context, create)


def _ast_expr_builder(
    func: t.Callable[P, Expr],
) -> t.Callable[P, BaseASTExpressionBuilder]:
    @wraps(func)
    def wrapper(self: _BaseBuilder, *args: P.args, **kwargs: P.kwargs) -> BaseASTExpressionBuilder:
        assert isinstance(self, _BaseBuilder)
        return BaseASTExpressionBuilder(self._context, partial(func, self, *args, **kwargs))  # type: ignore[arg-type]

    return wrapper  # type: ignore[return-value]


def _ast_stmt_builder(
    func: t.Callable[P, ast.stmt],
) -> t.Callable[P, ast.stmt]:
    @wraps(func)
    def wrapper(self: _BaseBuilder, *args: P.args, **kwargs: P.kwargs) -> ast.stmt:
        assert isinstance(self, _BaseBuilder)

        node = func(self, *args, **kwargs)  # type: ignore[arg-type]
        self._context.append_body(node)

        return node

    return wrapper  # type: ignore[return-value]


class AttrASTBuilder(BaseASTExpressionBuilder):
    def __init__(self, context: BuildContext, head: t.Union[str, TypeRef], *tail: str) -> None:
        super().__init__(context, self.__create_expr)
        self.__head = head
        self.__tail = tail

    @cached_property
    def parts(self) -> t.Sequence[str]:
        if isinstance(self.__head, str):
            return self.__head, *self.__tail

        queue = deque([self.__head])
        parts = list[str]()

        while queue:
            item = queue.pop()

            if isinstance(item, ast.Attribute):
                queue.append(item.value)
                parts.append(item.attr)

            elif isinstance(item, ast.Name):
                parts.append(item.id)

            elif isinstance(item, AttrASTBuilder):
                parts.extend(reversed(item.parts))

        return *reversed(parts), *self.__tail

    def attr(self, *tail: str) -> Self:
        return self.__class__(self._context, self, *tail)

    def call(
        self,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> CallASTBuilder:
        return CallASTBuilder(context=self._context, func=self, args=args, kwargs=kwargs)

    def assign(self, value: Expr) -> ast.stmt:
        return self._scope.assign_stmt(self, value)

    def __create_expr(self) -> Expr:
        return self._normalize_expr(
            ast.Name(id=self.__head) if isinstance(self.__head, str) else self.__head,
            *self.__tail,
        )


# noinspection PyTypeChecker
class CallASTBuilder(BaseASTExpressionBuilder):
    def __init__(
        self,
        context: BuildContext,
        func: TypeRef,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> None:
        super().__init__(context, self.__create_expr)
        self.__func = func
        self.__args = list[Expr]()
        self.__kwargs = dict[str, Expr]()
        self.__is_awaited = False

        for arg in args or ():
            self.arg(arg)

        for name, kwarg in (kwargs or {}).items():
            self.kwarg(name, kwarg)

    def await_(self, *, is_awaited: bool = True) -> Self:
        self.__is_awaited = is_awaited
        return self

    def arg(self, expr: Expr) -> Self:
        self.__args.append(expr)
        return self

    def kwarg(self, name: str, expr: Expr) -> Self:
        self.__kwargs[name] = expr
        return self

    def attr(self, *tail: str) -> AttrASTBuilder:
        return AttrASTBuilder(self._context, self, *tail)

    def call(
        self,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> Self:
        return self.__class__(context=self._context, func=self, args=args, kwargs=kwargs)

    def __create_expr(self) -> Expr:
        node: ast.expr = ast.Call(
            func=self._normalize_expr(self.__func),
            args=[self._normalize_expr(arg) for arg in self.__args],
            keywords=[ast.keyword(arg=key, value=self._normalize_expr(kwarg)) for key, kwarg in self.__kwargs.items()],
            lineno=0,
        )

        if self.__is_awaited:
            node = ast.Await(value=node)

        return node


class ClassTypeRefBuilder(_BaseBuilder, ASTExpressionBuilder):
    def __init__(
        self,
        context: BuildContext,
        info: TypeInfo,
        transform: t.Optional[t.Callable[[BuildContext, TypeInfo], Expr]] = None,
    ) -> None:
        super().__init__(context)
        self.__info = info
        self.__transform: t.Callable[[BuildContext, TypeInfo], Expr] = (
            transform if transform is not None else self.__ident
        )

    @property
    def info(self) -> TypeInfo:
        return self.__info

    def optional(self) -> ClassTypeRefBuilder:
        def transform(
            inner: t.Callable[[BuildContext, TypeInfo], Expr],
            context: BuildContext,
            info: TypeInfo,
        ) -> Expr:
            return self._scope.optional_type(inner(context, info))

        return self.__wrap(transform)

    def collection(self) -> ClassTypeRefBuilder:
        def transform(
            inner: t.Callable[[BuildContext, TypeInfo], Expr],
            context: BuildContext,
            info: TypeInfo,
        ) -> Expr:
            return self._scope.collection_type(inner(context, info))

        return self.__wrap(transform)

    def set(self) -> ClassTypeRefBuilder:
        def transform(
            inner: t.Callable[[BuildContext, TypeInfo], Expr],
            context: BuildContext,
            info: TypeInfo,
        ) -> Expr:
            return self._scope.generic_type(set, inner(context, info))

        return self.__wrap(transform)

    def sequence(self, *, mutable: bool = False) -> ClassTypeRefBuilder:
        def transform(
            inner: t.Callable[[BuildContext, TypeInfo], Expr],
            context: BuildContext,
            info: TypeInfo,
        ) -> Expr:
            return self._scope.sequence_type(inner(context, info), mutable=mutable)

        return self.__wrap(transform)

    def list(self) -> ClassTypeRefBuilder:
        def transform(
            inner: t.Callable[[BuildContext, TypeInfo], Expr],
            context: BuildContext,
            info: TypeInfo,
        ) -> Expr:
            return self._scope.generic_type(list, inner(context, info))

        return self.__wrap(transform)

    def mapping_key(self, value: TypeRef, *, mutable: bool = False) -> ClassTypeRefBuilder:
        def transform(
            inner: t.Callable[[BuildContext, TypeInfo], Expr],
            context: BuildContext,
            info: TypeInfo,
        ) -> Expr:
            return self._scope.mapping_type(inner(context, info), value, mutable=mutable)

        return self.__wrap(transform)

    def dict_key(self, value: TypeRef) -> ClassTypeRefBuilder:
        def transform(
            inner: t.Callable[[BuildContext, TypeInfo], Expr],
            context: BuildContext,
            info: TypeInfo,
        ) -> Expr:
            return self._scope.generic_type(dict, inner(context, info), value)

        return self.__wrap(transform)

    def mapping_value(self, key: TypeRef, *, mutable: bool = False) -> ClassTypeRefBuilder:
        def transform(
            inner: t.Callable[[BuildContext, TypeInfo], Expr],
            context: BuildContext,
            info: TypeInfo,
        ) -> Expr:
            return self._scope.mapping_type(key, inner(context, info), mutable=mutable)

        return self.__wrap(transform)

    def dict_value(self, key: TypeRef) -> ClassTypeRefBuilder:
        def transform(
            inner: t.Callable[[BuildContext, TypeInfo], Expr],
            context: BuildContext,
            info: TypeInfo,
        ) -> Expr:
            return self._scope.generic_type(dict, key, inner(context, info))

        return self.__wrap(transform)

    def context_manager(self, *, is_async: bool = False) -> ClassTypeRefBuilder:
        def transform(
            inner: t.Callable[[BuildContext, TypeInfo], Expr],
            context: BuildContext,
            info: TypeInfo,
        ) -> Expr:
            return self._scope.context_manager_type(inner(context, info), is_async=is_async)

        return self.__wrap(transform)

    def iterator(self, *, is_async: bool = False) -> ClassTypeRefBuilder:
        def transform(
            inner: t.Callable[[BuildContext, TypeInfo], Expr],
            context: BuildContext,
            info: TypeInfo,
        ) -> Expr:
            return self._scope.iterable_type(inner(context, info), is_async=is_async)

        return self.__wrap(transform)

    def iterable(self, *, is_async: bool = False) -> ClassTypeRefBuilder:
        def transform(
            inner: t.Callable[[BuildContext, TypeInfo], Expr],
            context: BuildContext,
            info: TypeInfo,
        ) -> Expr:
            return self._scope.iterable_type(inner(context, info), is_async=is_async)

        return self.__wrap(transform)

    def attr(self, *tail: str) -> AttrASTBuilder:
        return AttrASTBuilder(self._context, self, *tail)

    def init(
        self,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> CallASTBuilder:
        return CallASTBuilder(self._context, self, args, kwargs)

    @override
    def build_expr(self) -> ast.expr:
        return self._normalize_expr(self.__transform(self._context, self.__info))

    def __ident(self, context: BuildContext, info: TypeInfo) -> Expr:
        return context.resolver.resolve_expr(info)

    def __wrap(
        self,
        transform: t.Callable[[t.Callable[[BuildContext, TypeInfo], Expr], BuildContext, TypeInfo], Expr],
    ) -> ClassTypeRefBuilder:
        return self.__class__(self._context, self.__info, partial(transform, self.__transform))


@dataclass(frozen=True)
class Comprehension:
    target: Expr
    items: Expr
    predicates: t.Sequence[Expr] = field(default_factory=list)
    is_async: bool = False


# noinspection PyTypeChecker
class ScopeASTBuilder(_BaseBuilder):
    def type_ref(self, origin: t.Union[TypeInfo, RuntimeType]) -> ClassTypeRefBuilder:
        return ClassTypeRefBuilder(
            context=self._context,
            info=origin
            if isinstance(origin, (NamedTypeInfo, LiteralTypeInfo))
            else self._context.inspector.inspect(origin),
        )

    @_ast_expr_builder
    def const(self, value: object) -> Expr:
        assert not isinstance(value, ast.AST)
        return ast.Constant(value=value)

    def none(self) -> Expr:
        return self.const(None)

    def ellipsis(self) -> Expr:
        return self.const(...)

    @_ast_expr_builder
    def compare_expr(self, left: Expr, tests: t.Sequence[tuple[ast.cmpop, Expr]]) -> Expr:
        return ast.Compare(
            left=self._normalize_expr(left),
            ops=[op for op, _ in tests],
            comparators=[self._normalize_expr(comp) for _, comp in tests],
        )

    @_ast_expr_builder
    def ternary_expr(
        self,
        body: Expr,
        test: Expr,
        or_else: Expr,
    ) -> Expr:
        return ast.IfExp(
            test=self._normalize_expr(test),
            body=self._normalize_expr(body),
            orelse=self._normalize_expr(or_else),
        )

    @_ast_expr_builder
    def ternary_not_none_expr(
        self,
        body: Expr,
        test: Expr,
        or_else: t.Optional[Expr] = None,
    ) -> Expr:
        return self.ternary_expr(
            test=self.compare_is_not_expr(test, self.none()),
            body=body,
            or_else=or_else if or_else is not None else self.none(),
        )

    @_ast_expr_builder
    def tuple_expr(self, *items: TypeRef, normalize: bool = False) -> Expr:
        if normalize and len(items) == 1:
            return self._normalize_expr(items[0])

        return ast.Tuple(elts=[self._normalize_expr(item) for item in items])

    @t.overload
    def set_expr(self, items: t.Union[Comprehension, t.Sequence[Comprehension]], element: Expr) -> Expr: ...

    @t.overload
    def set_expr(self, items: t.Collection[Expr]) -> Expr: ...

    @_ast_expr_builder
    def set_expr(
        self,
        items: t.Union[Comprehension, t.Sequence[Comprehension], t.Collection[Expr]],
        element: t.Optional[Expr] = None,
    ) -> Expr:
        if self.__is_comprehensions(items):
            assert element is not None

            return ast.SetComp(
                elt=self._normalize_expr(element),
                generators=self.__build_comprehensions(items),
            )

        elif self.__is_expression_collection(items):
            return ast.Set(elts=[self._normalize_expr(item) for item in items])

        else:
            # TODO: make assert_never work
            raise RuntimeError(items)

    @t.overload
    def list_expr(self, items: t.Union[Comprehension, t.Sequence[Comprehension]], element: Expr) -> Expr: ...

    @t.overload
    def list_expr(self, items: t.Sequence[Expr]) -> Expr: ...

    @_ast_expr_builder
    def list_expr(
        self,
        items: t.Union[Comprehension, t.Sequence[Comprehension], t.Sequence[Expr]],
        element: t.Optional[Expr] = None,
    ) -> Expr:
        if self.__is_comprehensions(items):
            assert element is not None

            return ast.ListComp(
                elt=self._normalize_expr(element),
                generators=self.__build_comprehensions(items),
            )

        elif self.__is_expression_sequence(items):
            return ast.List(elts=[self._normalize_expr(item) for item in items])

        else:
            # TODO: make assert_never work
            raise RuntimeError(items)

    @t.overload
    def dict_expr(
        self,
        items: t.Union[Comprehension, t.Sequence[Comprehension]],
        key: Expr,
        value: Expr,
    ) -> Expr: ...

    @t.overload
    def dict_expr(self, items: t.Mapping[Expr, Expr]) -> Expr: ...

    @_ast_expr_builder
    def dict_expr(
        self,
        items: t.Union[Comprehension, t.Sequence[Comprehension], t.Mapping[Expr, Expr]],
        key: t.Optional[Expr] = None,
        value: t.Optional[Expr] = None,
    ) -> Expr:
        if self.__is_comprehensions(items):
            assert key is not None
            assert value is not None

            return ast.DictComp(
                key=self._normalize_expr(key),
                value=self._normalize_expr(value),
                generators=self.__build_comprehensions(items),
            )

        elif self.__is_expression_mapping(items):
            keys = list[t.Optional[ast.expr]]()
            values = list[ast.expr]()

            for k, v in items.items():
                keys.append(self._normalize_expr(k))
                values.append(self._normalize_expr(v))

            return ast.Dict(keys=keys, values=values)

        else:
            # TODO: make assert_never work
            raise RuntimeError(items)

    @_ast_expr_builder
    def not_op(self, expr: Expr) -> Expr:
        return ast.UnaryOp(op=ast.Not(), operand=self._normalize_expr(expr))

    @_ast_expr_builder
    def compare_is_expr(self, left: Expr, right: Expr) -> Expr:
        return ast.Compare(
            left=self._normalize_expr(left),
            ops=[ast.Is()],
            comparators=[self._normalize_expr(right)],
        )

    @_ast_expr_builder
    def compare_is_not_expr(self, left: Expr, right: Expr) -> Expr:
        return ast.Compare(
            left=self._normalize_expr(left),
            ops=[ast.IsNot()],
            comparators=[self._normalize_expr(right)],
        )

    @_ast_expr_builder
    def compare_eq_expr(self, left: Expr, right: Expr) -> Expr:
        return ast.Compare(
            left=self._normalize_expr(left),
            ops=[ast.Eq()],
            comparators=[self._normalize_expr(right)],
        )

    @_ast_expr_builder
    def compare_not_eq_expr(self, left: Expr, right: Expr) -> Expr:
        return ast.Compare(
            left=self._normalize_expr(left),
            ops=[ast.NotEq()],
            comparators=[self._normalize_expr(right)],
        )

    @_ast_expr_builder
    def compare_lt_expr(self, left: Expr, right: Expr) -> Expr:
        return ast.Compare(
            left=self._normalize_expr(left),
            ops=[ast.Lt()],
            comparators=[self._normalize_expr(right)],
        )

    @_ast_expr_builder
    def compare_lte_expr(self, left: Expr, right: Expr) -> Expr:
        return ast.Compare(
            left=self._normalize_expr(left),
            ops=[ast.LtE()],
            comparators=[self._normalize_expr(right)],
        )

    @_ast_expr_builder
    def compare_gt_expr(self, left: Expr, right: Expr) -> Expr:
        return ast.Compare(
            left=self._normalize_expr(left),
            ops=[ast.Gt()],
            comparators=[self._normalize_expr(right)],
        )

    @_ast_expr_builder
    def compare_gte_expr(self, left: Expr, right: Expr) -> Expr:
        return ast.Compare(
            left=self._normalize_expr(left),
            ops=[ast.GtE()],
            comparators=[self._normalize_expr(right)],
        )

    def attr(self, head: t.Union[str, TypeRef], *tail: str) -> AttrASTBuilder:
        return AttrASTBuilder(self._context, head, *tail)

    def call(
        self,
        func: TypeRef,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> CallASTBuilder:
        return CallASTBuilder(self._context, func, args, kwargs)

    @_ast_expr_builder
    def generic_type(self, generic: TypeRef, *args: TypeRef) -> Expr:
        if len(args) == 0:
            return self._normalize_expr(generic)

        if len(args) == 1:
            return ast.Subscript(
                value=self._normalize_expr(generic),
                slice=self._normalize_expr(args[0]),
            )

        return ast.Subscript(
            value=self._normalize_expr(generic),
            slice=self._normalize_expr(self.tuple_expr(*args)),
        )

    def literal_type(self, *args: t.Union[str, Expr]) -> Expr:
        if not args:
            return self._normalize_expr(predef().no_return)

        return self.generic_type(
            predef().literal,
            *(self.const(arg) if isinstance(arg, str) else arg for arg in args),
        )

    def optional_type(self, of_type: TypeRef) -> Expr:
        return self.generic_type(predef().optional, of_type)

    def collection_type(self, of_type: TypeRef) -> Expr:
        return self.generic_type(predef().collection, of_type)

    def sequence_type(self, of_type: TypeRef, *, mutable: bool = False) -> Expr:
        return self.generic_type(predef().mutable_sequence if mutable else predef().sequence, of_type)

    def mapping_type(self, key_type: TypeRef, value_type: TypeRef, *, mutable: bool = False) -> Expr:
        return self.generic_type(predef().mutable_mapping if mutable else predef().mapping, key_type, value_type)

    def iterator_type(self, of_type: TypeRef, *, is_async: bool = False) -> Expr:
        return self.generic_type(predef().async_iterator if is_async else predef().iterator, of_type)

    def iterable_type(self, of_type: TypeRef, *, is_async: bool = False) -> Expr:
        return self.generic_type(predef().async_iterable if is_async else predef().iterable, of_type)

    def context_manager_type(self, of_type: TypeRef, *, is_async: bool = False) -> Expr:
        return self.generic_type(
            predef().async_context_manager if is_async else predef().context_manager,
            of_type,
        )

    def stmt(self, *stmts: t.Optional[Stmt]) -> None:
        self._context.extend_body(self._context.resolver.resolve_stmts(*stmts))

    def class_def(self, name: str) -> ClassStatementASTBuilder:
        return ClassStatementASTBuilder(self._context, name)

    def func_def(self, name: str) -> FuncStatementASTBuilder:
        return FuncStatementASTBuilder(self._context, name)

    @_ast_stmt_builder
    def field_def(self, name: str, annotation: TypeRef, default: t.Optional[Expr] = None) -> ast.stmt:
        return ast.AnnAssign(
            target=ast.Name(id=name),
            annotation=self._normalize_expr(annotation),
            value=self._normalize_expr(default) if default is not None else None,
            simple=1,
        )

    @_ast_stmt_builder
    def assign_stmt(self, target: t.Union[str, Expr], value: Expr) -> ast.stmt:
        return ast.Assign(
            targets=[self._normalize_expr(self.attr(target))],
            value=self._normalize_expr(value),
            lineno=0,
        )

    def if_stmt(self, test: Expr) -> IfStatementASTBuilder:
        return IfStatementASTBuilder(self._context, test)

    def for_stmt(self, target: str, items: Expr) -> ForStatementASTBuilder:
        return ForStatementASTBuilder(self._context, target, items)

    def while_stmt(self, test: Expr) -> WhileStatementASTBuilder:
        return WhileStatementASTBuilder(self._context, test)

    @_ast_stmt_builder
    def break_stmt(self) -> ast.stmt:
        return ast.Break()

    @_ast_stmt_builder
    def continue_stmt(self) -> ast.stmt:
        return ast.Continue()

    @_ast_stmt_builder
    def return_stmt(self, value: Expr) -> ast.stmt:
        return ast.Return(
            value=self._normalize_expr(value),
            lineno=0,
        )

    @_ast_stmt_builder
    def yield_stmt(self, value: Expr) -> ast.stmt:
        return ast.Expr(
            value=ast.Yield(
                value=self._normalize_expr(value),
                lineno=0,
            ),
        )

    def try_stmt(self) -> TryStatementASTBuilder:
        return TryStatementASTBuilder(self._context)

    @_ast_stmt_builder
    def raise_stmt(self, err: Expr, cause: t.Optional[Expr] = None) -> ast.stmt:
        return ast.Raise(
            exc=self._normalize_expr(err),
            cause=self._normalize_expr(cause) if cause is not None else None,
        )

    def with_stmt(self) -> WithStatementASTBuilder:
        return WithStatementASTBuilder(self._context)

    @_ast_stmt_builder
    def ellipsis_stmt(self) -> ast.stmt:
        return ast.Expr(value=ast.Constant(value=...))

    @_ast_stmt_builder
    def pass_stmt(self) -> ast.stmt:
        return ast.Pass()

    def __is_comprehensions(self, obj: object) -> TypeGuard[t.Union[Comprehension, t.Sequence[Comprehension]]]:
        return isinstance(obj, Comprehension) or (
            isinstance(obj, t.Sequence) and all(isinstance(item, Comprehension) for item in obj)
        )

    def __is_expression_collection(self, obj: object) -> TypeGuard[t.Collection[Expr]]:
        return isinstance(obj, t.Collection) and all(isinstance(item, (ast.expr, ASTExpressionBuilder)) for item in obj)

    def __is_expression_sequence(self, obj: object) -> TypeGuard[t.Sequence[Expr]]:
        return isinstance(obj, t.Sequence) and all(isinstance(item, (ast.expr, ASTExpressionBuilder)) for item in obj)

    def __is_expression_mapping(self, obj: object) -> TypeGuard[t.Mapping[Expr, Expr]]:
        return isinstance(obj, t.Mapping) and all(
            isinstance(item, (ast.expr, ASTExpressionBuilder)) for pair in obj.items() for item in pair
        )

    def __build_comprehensions(
        self,
        comprehensions: t.Union[Comprehension, t.Sequence[Comprehension]],
    ) -> list[ast.comprehension]:
        return [
            ast.comprehension(
                target=self._normalize_expr(compr.target),
                iter=self._normalize_expr(compr.items),
                ifs=[self._normalize_expr(predicate) for predicate in compr.predicates or ()],
                is_async=compr.is_async,
            )
            for compr in ([comprehensions] if isinstance(comprehensions, Comprehension) else comprehensions)
        ]


class _NestedBlockASTBuilder(_BaseBuilder, ASTStatementBuilder, metaclass=abc.ABCMeta):
    def __init__(self, context: BuildContext, *, allow_implicit_enter: bool = True) -> None:
        super().__init__(context)
        self.__entered = False
        self.__allow_implicit_enter = allow_implicit_enter

    def __enter__(self) -> Self:
        self.__entered = True
        return self

    def __exit__(self, exc_type: object, exc_value: object, exc_traceback: object) -> None:
        self.__entered = False

        if exc_type is None:
            self._context.extend_body(self.build_stmt())

    def _block(self, body: list[ast.stmt]) -> t.ContextManager[ScopeASTBuilder]:
        return self.__enter_block(body) if self.__entered else self.__enter_implicitly(body)

    @contextmanager
    def __enter_block(self, body: list[ast.stmt]) -> t.Iterator[ScopeASTBuilder]:
        self._context.enter_scope(None, body)
        yield ScopeASTBuilder(self._context)
        self._context.leave_scope()

    @contextmanager
    def __enter_implicitly(self, body: list[ast.stmt]) -> t.Iterator[ScopeASTBuilder]:
        if not self.__allow_implicit_enter:
            msg = "can't enter into the nested block implicitly"
            raise RuntimeError(msg, self, body)

        with self, self.__enter_block(body) as scope:
            yield scope


# noinspection PyTypeChecker
class WhileStatementASTBuilder(_NestedBlockASTBuilder):
    def __init__(self, context: BuildContext, test: Expr) -> None:
        super().__init__(context)
        self.__test = test
        self.__body = list[ast.stmt]()
        self.__else = list[ast.stmt]()

    def body(self) -> t.ContextManager[ScopeASTBuilder]:
        return self._block(self.__body)

    def else_(self) -> t.ContextManager[ScopeASTBuilder]:
        return self._block(self.__else)

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        return [
            ast.While(
                test=self._normalize_expr(self.__test),
                body=self._normalize_body(self.__body),
                orelse=self.__else,
                lineno=0,
            ),
        ]


# noinspection PyTypeChecker
class ForStatementASTBuilder(_NestedBlockASTBuilder):
    def __init__(self, context: BuildContext, target: str, items: Expr) -> None:
        super().__init__(context)
        self.__target = target
        self.__items = items
        self.__body = list[ast.stmt]()
        self.__else = list[ast.stmt]()
        self.__is_async = False

    def async_(self, *, is_async: bool = True) -> Self:
        self.__is_async = is_async
        return self

    def body(self) -> t.ContextManager[ScopeASTBuilder]:
        return self._block(self.__body)

    def else_(self) -> t.ContextManager[ScopeASTBuilder]:
        return self._block(self.__else)

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        return [
            ast.AsyncFor(
                target=ast.Name(id=self.__target),
                iter=self._normalize_expr(self.__items),
                body=self._normalize_body(self.__body),
                orelse=self.__else,
                lineno=0,
            )
            if self.__is_async
            else ast.For(
                target=ast.Name(id=self.__target),
                iter=self._normalize_expr(self.__items),
                body=self._normalize_body(self.__body),
                orelse=self.__else,
                lineno=0,
            ),
        ]


# noinspection PyTypeChecker
class WithStatementASTBuilder(_NestedBlockASTBuilder):
    def __init__(self, context: BuildContext) -> None:
        super().__init__(context)
        self.__cms = list[tuple[Expr, t.Optional[str]]]()
        self.__body = list[ast.stmt]()
        self.__is_async = False

    def async_(self, *, is_async: bool = True) -> Self:
        self.__is_async = is_async
        return self

    def enter(self, cm: Expr, name: t.Optional[str] = None) -> Self:
        self.__cms.append((cm, name))
        return self

    def body(self) -> t.ContextManager[ScopeASTBuilder]:
        return self._block(self.__body)

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        if not self.__cms:
            msg = "with statement must have at least one expression"
            raise IncompleteStatementError(msg, self)

        items = [
            ast.withitem(
                context_expr=self._normalize_expr(cm),
                optional_vars=ast.Name(id=name) if name is not None else None,
            )
            for cm, name in self.__cms
        ]

        return [
            ast.AsyncWith(
                items=items,
                body=self._normalize_body(self.__body),
                lineno=0,
            )
            if self.__is_async
            else ast.With(
                items=items,
                body=self._normalize_body(self.__body),
                lineno=0,
            ),
        ]


# noinspection PyTypeChecker
class IfStatementASTBuilder(_NestedBlockASTBuilder):
    def __init__(self, context: BuildContext, test: Expr) -> None:
        super().__init__(context)
        self.__test = test
        self.__body = list[ast.stmt]()
        self.__else = list[ast.stmt]()

    def body(self) -> t.ContextManager[ScopeASTBuilder]:
        return self._block(self.__body)

    def else_(self) -> t.ContextManager[ScopeASTBuilder]:
        return self._block(self.__else)

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        return [
            ast.If(
                test=self._normalize_expr(self.__test),
                body=self._normalize_body(self.__body),
                orelse=self.__else,
                lineno=0,
            ),
        ]


# noinspection PyTypeChecker
class TryStatementASTBuilder(_NestedBlockASTBuilder):
    @dataclass()
    class ExceptHandler:
        types: t.Sequence[TypeRef]
        name: t.Optional[str]
        body: list[ast.stmt]

    def __init__(self, context: BuildContext) -> None:
        super().__init__(context, allow_implicit_enter=False)
        self.__body = list[ast.stmt]()
        self.__handlers = list[TryStatementASTBuilder.ExceptHandler]()
        self.__else = list[ast.stmt]()
        self.__finally = list[ast.stmt]()

    def body(self) -> t.ContextManager[ScopeASTBuilder]:
        return self._block(self.__body)

    def except_(self, *types: TypeRef, name: t.Optional[str] = None) -> t.ContextManager[ScopeASTBuilder]:
        body = list[ast.stmt]()

        self.__handlers.append(
            self.ExceptHandler(
                types=types,
                name=name,
                body=body,
            )
        )

        return self._block(body)

    def else_(self) -> t.ContextManager[ScopeASTBuilder]:
        return self._block(self.__else)

    def finally_(self) -> t.ContextManager[ScopeASTBuilder]:
        return self._block(self.__finally)

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        if not self.__handlers and not self.__finally:
            msg = "try statement must have at least one `except` or `finally` block"
            raise IncompleteStatementError(msg, self)

        scope = self._scope

        return [
            ast.Try(
                body=self.__body,
                handlers=[
                    ast.ExceptHandler(
                        type=self._normalize_expr(scope.tuple_expr(*handler.types, normalize=True)),
                        name=handler.name,
                        body=handler.body or [ast.Pass()],
                    )
                    for handler in self.__handlers
                ],
                orelse=self.__else,
                finalbody=self.__finally,
            ),
        ]


@dataclass(frozen=True)
class TypeVar:
    name: str
    bound: t.Optional[TypeRef] = None


class ClassScopeASTBuilder(ScopeASTBuilder, TypeDefinitionBuilder):
    def __init__(self, context: BuildContext, header: ClassStatementASTBuilder) -> None:
        super().__init__(context)
        self.__header = header

    @override
    @property
    def info(self) -> TypeInfo:
        return self.__header.info

    @override
    def ref(self) -> ClassTypeRefBuilder:
        return self.__header.ref()

    def method_def(self, name: str) -> MethodStatementASTBuilder:
        return MethodStatementASTBuilder(self._context, name)

    def new_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__new__")

    def init_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__init__").returns(self.const(None))

    @contextmanager
    def init_self_attrs_def(self, attrs: t.Mapping[str, TypeRef]) -> t.Iterator[MethodScopeASTBuilder]:
        init_def = self.init_def()

        for name, annotation in attrs.items():
            init_def.arg(name=name, annotation=annotation)

        with init_def as init_body:
            for name in attrs:
                init_body.assign_stmt(init_body.self_attr(name), init_body.attr(name))

            yield init_body

    def call_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__call__")

    def str_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__str__").returns(str)

    def repr_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__repr__").returns(str)

    def hash_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__hash__").returns(bool)

    def eq_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__eq__").returns(bool)

    def ne_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__ne__").returns(bool)

    def ge_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__ge__").returns(bool)

    def gt_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__gt__").returns(bool)

    def le_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__le__").returns(bool)

    def lt_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__lt__").returns(bool)

    def property_getter_def(self, name: str) -> FuncStatementASTBuilder:
        return self.func_def(name).arg("self").decorators(predef().property)

    def property_setter_def(self, name: str) -> FuncStatementASTBuilder:
        return self.func_def(name).arg("self").decorators(self.attr(name, "setter"))

    def property_deleter_def(self, name: str) -> FuncStatementASTBuilder:
        return self.func_def(name).arg("self").decorators(self.attr(name, "deleter"))


# noinspection PyTypeChecker
class ClassStatementASTBuilder(
    _BaseBuilder,
    t.ContextManager[ClassScopeASTBuilder],
    ASTStatementBuilder,
    TypeDefinitionBuilder,
):
    def __init__(self, context: BuildContext, name: str) -> None:
        super().__init__(context)
        self.__info = NamedTypeInfo(name=name, module=self._context.module, namespace=self._context.namespace)
        self.__bases = list[TypeRef]()
        self.__decorators = list[TypeRef]()
        self.__keywords = dict[str, TypeRef]()
        self.__type_vars = list[TypeVar]()
        self.__docs = list[str]()
        self.__body = list[ast.stmt]()

    @override
    def __enter__(self) -> ClassScopeASTBuilder:
        self._context.enter_scope(self.__info.name, self.__body)
        return ClassScopeASTBuilder(self._context, self)

    @override
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if exc_type is None:
            self._context.leave_scope()
            self._context.extend_body(self.build_stmt())

    @override
    @property
    def info(self) -> TypeInfo:
        return self.__info

    @override
    def ref(self) -> ClassTypeRefBuilder:
        return ClassTypeRefBuilder(self._context, self.__info)

    if sys.version_info >= (3, 12):

        def type_param(self, name: str, bound: t.Optional[TypeRef] = None) -> Self:
            self.__type_vars.append(TypeVar(name=name, bound=bound))
            return self

    def docstring(self, value: t.Optional[str]) -> Self:
        if value:
            self.__docs.append(value)

        return self

    def abstract(self) -> Self:
        return self.keywords(metaclass=predef().abc_meta)

    def dataclass(self, *, frozen: bool = False, kw_only: bool = False) -> Self:
        dc = CallASTBuilder(self._context, predef().dataclass_decorator)

        if frozen:
            dc.kwarg("frozen", ast.Constant(value=frozen))

        if kw_only:
            if sys.version_info < (3, 10):
                warnings.warn("`kw_only` is not supported by current python version", UserWarning, stacklevel=2)

            dc.kwarg("kw_only", ast.Constant(value=kw_only))

        return self.decorators(dc)

    def inherits(self, *bases: t.Optional[TypeRef]) -> Self:
        self.__bases.extend(base for base in bases if base is not None)
        return self

    def decorators(self, *items: t.Optional[TypeRef]) -> Self:
        self.__decorators.extend(item for item in items if item is not None)
        return self

    def keywords(self, **keywords: t.Optional[TypeRef]) -> Self:
        self.__keywords.update({key: value for key, value in keywords.items() if value is not None})
        return self

    # NOTE: workaround for passing mypy typings in CI for python 3.12
    if sys.version_info >= (3, 12):
        # noinspection PyArgumentList
        @override
        def build_stmt(self) -> t.Sequence[ast.stmt]:
            return [
                ast.ClassDef(
                    name=self.__info.name,
                    bases=self.__build_bases(),
                    keywords=self.__build_keywords(),
                    body=self._normalize_body(self.__body, self.__docs),
                    decorator_list=self.__build_decorators(),
                    type_params=[
                        ast.TypeVar(
                            name=type_var.name,
                            bound=self._normalize_expr(type_var.bound) if type_var.bound is not None else None,
                        )
                        for type_var in self.__type_vars
                    ],
                ),
            ]

    else:
        # noinspection PyArgumentList
        @override
        def build_stmt(self) -> t.Sequence[ast.stmt]:
            return [
                ast.ClassDef(
                    name=self.__info.name,
                    bases=self.__build_bases(),
                    keywords=self.__build_keywords(),
                    body=self._normalize_body(self.__body, self.__docs),
                    decorator_list=self.__build_decorators(),
                ),
            ]

    def __build_bases(self) -> list[ast.expr]:
        return [self._normalize_expr(base) for base in self.__bases]

    def __build_keywords(self) -> list[ast.keyword]:
        return [ast.keyword(arg=key, value=self._normalize_expr(value)) for key, value in self.__keywords.items()]

    def __build_decorators(self) -> list[ast.expr]:
        return [self._normalize_expr(dec) for dec in self.__decorators]


class FuncTypeRefBuilder(ASTExpressionBuilder):
    def __init__(self, context: BuildContext, info: NamedTypeInfo) -> None:
        self.__context = context
        self.__info = info

    @override
    def build_expr(self) -> ast.expr:
        return self.__context.resolver.resolve_expr(self.__info)


# noinspection PyTypeChecker
class FuncStatementASTBuilder(
    _BaseBuilder,
    t.ContextManager[ScopeASTBuilder],
    ASTStatementBuilder,
    TypeDefinitionBuilder,
):
    def __init__(self, context: BuildContext, name: str) -> None:
        super().__init__(context)
        self.__info = NamedTypeInfo(name=name, module=self._context.module, namespace=self._context.namespace)
        self.__decorators = list[TypeRef]()
        self.__args = list[tuple[str, t.Optional[TypeRef]]]()
        self.__kwargs = dict[str, t.Optional[TypeRef]]()
        self.__defaults = dict[str, Expr]()
        self.__returns: t.Optional[TypeRef] = None
        self.__is_async = False
        self.__is_abstract = False
        self.__is_override = False
        self.__iterator_cm = False
        self.__is_stub = False
        self.__is_not_implemented = False
        self.__docs = list[str]()
        self.__body = list[ast.stmt]()

    @override
    def __enter__(self) -> ScopeASTBuilder:
        self._context.enter_scope(self.__info.name, self.__body)
        return ScopeASTBuilder(self._context)

    @override
    def __exit__(self, exc_type: object, exc_value: object, exc_traceback: object) -> None:
        if exc_type is None:
            self._context.leave_scope()
            self._context.extend_body(self.build_stmt())

    @override
    @property
    def info(self) -> TypeInfo:
        return self.__info

    @override
    def ref(self) -> FuncTypeRefBuilder:
        return FuncTypeRefBuilder(self._context, self.__info)

    def async_(self, *, is_async: bool = True) -> Self:
        self.__is_async = is_async
        return self

    def abstract(self) -> Self:
        self.__is_abstract = True
        return self

    def overrides(self) -> Self:
        self.__is_override = True
        return self

    def docstring(self, value: t.Optional[str]) -> Self:
        if value:
            self.__docs.append(value)
        return self

    def decorators(self, *items: t.Optional[TypeRef]) -> Self:
        self.__decorators.extend(item for item in items if item is not None)
        return self

    def arg(
        self,
        name: str,
        annotation: t.Optional[TypeRef] = None,
        default: t.Optional[Expr] = None,
    ) -> Self:
        self.__args.append((name, annotation))

        if default is not None:
            self.__defaults[name] = default

        return self

    def returns(self, ret: t.Optional[TypeRef]) -> Self:
        if ret is not None:
            self.__returns = ret
        return self

    def context_manager(self) -> Self:
        self.__iterator_cm = True
        return self

    def stub(self) -> Self:
        self.__is_stub = True
        return self

    def not_implemented(self) -> Self:
        self.__is_not_implemented = True
        return self

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        node: ast.stmt

        scope = ScopeASTBuilder(self._context)

        if self.__is_async:
            # noinspection PyArgumentList
            node = ast.AsyncFunctionDef(  # type: ignore[call-overload,no-any-return,unused-ignore]
                # type_comment and type_params has default value each in 3.12 and not available in 3.9
                name=self.__info.name,
                decorator_list=self.__build_decorators(),
                args=self.__build_args(),
                returns=self.__build_returns(scope),
                body=self.__build_body(),
                lineno=0,
            )

        else:
            # noinspection PyArgumentList
            node = ast.FunctionDef(  # type: ignore[call-overload,no-any-return,unused-ignore]
                # type_comment and type_params has default value each in 3.12 and not available in 3.9
                name=self.__info.name,
                decorator_list=self.__build_decorators(),
                args=self.__build_args(),
                returns=self.__build_returns(scope),
                body=self.__build_body(),
                lineno=0,
            )

        return [node]

    def __build_decorators(self) -> list[ast.expr]:
        head_decorators: list[TypeRef] = []
        last_decorators: list[TypeRef] = []

        if self.__is_override:
            head_decorators.append(predef().override_decorator)

        if self.__is_abstract:
            last_decorators.append(predef().abstractmethod)

        if self.__iterator_cm:
            last_decorators.append(
                predef().async_context_manager_decorator if self.__is_async else predef().context_manager_decorator
            )

        return [self._normalize_expr(dec) for dec in chain(head_decorators, self.__decorators, last_decorators)]

    def __build_args(self) -> ast.arguments:
        return ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(
                    arg=arg,
                    annotation=self._normalize_expr(annotation) if annotation is not None else None,
                )
                for arg, annotation in self.__args
            ],
            defaults=[self._normalize_expr(self.__defaults[arg]) for arg, _ in self.__args if arg in self.__defaults],
            kwonlyargs=[
                ast.arg(
                    arg=arg,
                    annotation=self._normalize_expr(annotation) if annotation is not None else None,
                )
                for arg, annotation in self.__kwargs.items()
            ],
            kw_defaults=[self._normalize_expr(self.__defaults[key]) for key in self.__kwargs if key in self.__defaults],
        )

    def __build_returns(self, scope: ScopeASTBuilder) -> t.Optional[ast.expr]:
        if self.__returns is None:
            return None

        ret = self.__returns
        if self.__iterator_cm:
            ret = scope.iterator_type(ret, is_async=self.__is_async)

        return self._normalize_expr(ret)

    def __build_body(self) -> list[ast.stmt]:
        body: t.Sequence[Stmt]

        if self.__is_stub:
            body = [ast.Expr(value=ast.Constant(value=...))]

        elif self.__is_not_implemented:
            body = [ast.Raise(exc=ast.Name(id="NotImplementedError"))]

        else:
            body = self.__body

        return self._normalize_body(body, self.__docs)


class MethodScopeASTBuilder(ScopeASTBuilder):
    def self_attr(
        self,
        head: str,
        *tail: str,
        mode: t.Literal["private", "protected", "public"] = "private",
    ) -> AttrASTBuilder:
        return self.attr(
            "self",
            f"__{head}" if mode == "private" else f"_{head}" if mode == "protected" else head,
            *tail,
        )


class MethodStatementASTBuilder(FuncStatementASTBuilder):
    def __init__(self, context: BuildContext, name: str) -> None:
        super().__init__(context, name)
        self.arg("self")

    @override
    def __enter__(self) -> MethodScopeASTBuilder:
        super().__enter__()
        return MethodScopeASTBuilder(self._context)


class ModuleASTBuilder(t.ContextManager["ModuleASTBuilder"], ScopeASTBuilder):
    def __init__(self, context: BuildContext, info: ModuleInfo, body: list[ast.stmt]) -> None:
        super().__init__(context)
        self.__info = info
        self.__body = body
        self.__docs = list[str]()

    @override
    def __enter__(self) -> Self:
        self._context.enter_module(self.__info, self.__body)
        return self

    @override
    def __exit__(self, exc_type: object, exc_value: object, exc_traceback: object) -> None:
        if exc_type is None:
            scope = self._context.leave_module()
            assert scope.body is self.__body

    @property
    def info(self) -> ModuleInfo:
        return self.__info

    def docstring(self, value: t.Optional[str]) -> Self:
        if value:
            self.__docs.append(value)
        return self

    def import_stmt(self, info: ModuleInfo) -> ast.Import:
        return ast.Import(names=[ast.alias(name=info.qualname)])

    def from_import_stmt(self, info: ModuleInfo, *names: str) -> ast.ImportFrom:
        return ast.ImportFrom(module=info.qualname, names=[ast.alias(name=name) for name in names], level=0)

    def build(self) -> ast.Module:
        return ast.Module(
            body=self._context.resolver.resolve_stmts(*self.__build_imports(), *self.__body, docs=self.__docs),
            type_ignores=[],
        )

    def render(self) -> str:
        return render_module(self.build())

    def write(self, *, mode: t.Literal["w", "a"] = "w", mkdir: bool = False, exist_ok: bool = False) -> None:
        write_module(self.build(), self.__info.file, mode=mode, mkdir=mkdir, exist_ok=exist_ok)

    def __build_imports(self) -> t.Sequence[ast.stmt]:
        return [
            self.import_stmt(dep)
            for dep in sorted(self._context.get_dependencies(self.__info), key=self.__get_dep_sort_key)
        ]

    def __get_dep_sort_key(self, info: ModuleInfo) -> str:
        return info.qualname


class PackageASTBuilder(t.ContextManager["PackageASTBuilder"]):
    def __init__(
        self,
        context: BuildContext,
        info: PackageInfo,
        modules: dict[ModuleInfo, ModuleASTBuilder],
    ) -> None:
        self.__context = context
        self.__info = info
        self.__modules = modules

    @override
    def __enter__(self) -> Self:
        self.__context.enter_package(self.__info)
        return self

    @override
    def __exit__(self, exc_type: object, exc_value: object, exc_traceback: object) -> None:
        if exc_type is None:
            self.__context.leave_package()

    @property
    def info(self) -> PackageInfo:
        return self.__info

    def sub(self, name: str) -> Self:
        return self.__class__(self.__context, PackageInfo(name, self.__info), self.__modules)

    def init(self) -> ModuleASTBuilder:
        return self.module("__init__")

    def module(self, name: str) -> ModuleASTBuilder:
        info = ModuleInfo(name, self.__info)

        builder = self.__modules.get(info)
        if builder is None:
            builder = self.__modules[info] = ModuleASTBuilder(self.__context, info, [])

        return builder

    def iter_modules(self) -> t.Iterable[ModuleASTBuilder]:
        for builder in self.__modules.values():
            if builder.info.qualname.startswith(self.__info.qualname):
                yield builder

    def build(self) -> t.Iterable[tuple[ModuleInfo, ast.Module]]:
        for builder in self.iter_modules():
            yield builder.info, builder.build()

    def render(self) -> t.Iterable[tuple[ModuleInfo, str]]:
        for builder in self.iter_modules():
            yield builder.info, builder.render()

    def write(self, *, mode: t.Literal["w", "a"] = "w", mkdir: bool = False, exist_ok: bool = False) -> None:
        for builder in self.iter_modules():
            builder.write(mode=mode, mkdir=mkdir, exist_ok=exist_ok)
