from __future__ import annotations

__all__ = [
    "AttrASTBuilder",
    "CallASTBuilder",
    "ClassScopeASTBuilder",
    "ClassStatementASTBuilder",
    "Comprehension",
    "ForStatementASTBuilder",
    "FuncArgInfo",
    "FuncStatementASTBuilder",
    "IfStatementASTBuilder",
    "MethodScopeASTBuilder",
    "MethodStatementASTBuilder",
    "ModuleASTBuilder",
    "PackageASTBuilder",
    "ScopeASTBuilder",
    "TryStatementASTBuilder",
    "TypeRefBuilder",
    "WhileStatementASTBuilder",
    "WithStatementASTBuilder",
    "build_module",
    "build_package",
]

import abc
import ast
import sys
import typing as t
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property, partial, wraps
from itertools import chain

from typing_extensions import ParamSpec

from astlab._typing import EllipsisType, Self, TypeGuard, assert_never, override
from astlab.abc import (
    ASTExpressionBuilder,
    ASTLabError,
    ASTResolver,
    ASTStatementBuilder,
    Expr,
    Stmt,
    TypeDefinitionBuilder,
    TypeExpr,
    TypeRef,
)
from astlab.context import BuildContext
from astlab.resolver import DefaultASTResolver
from astlab.types import (
    EnumTypeInfo,
    LiteralTypeInfo,
    ModuleInfo,
    NamedTypeInfo,
    PackageInfo,
    TypeInfo,
    TypeInspector,
    TypeVarInfo,
    TypeVarVariance,
    UnionTypeInfo,
    predef,
)
from astlab.version import PythonVersion
from astlab.writer import render_module, write_module

T_co = t.TypeVar("T_co", covariant=True)
P = ParamSpec("P")


def build_package(
    info: t.Union[str, PackageInfo],
    parent: t.Optional[PackageInfo] = None,
    resolver: t.Optional[ASTResolver] = None,
    inspector: t.Optional[TypeInspector] = None,
    python_version: t.Union[PythonVersion, t.Sequence[int], None] = None,
) -> PackageASTBuilder:
    """Start python package builder."""
    return PackageASTBuilder(
        context=_create_context(resolver, inspector, python_version),
        info=info if isinstance(info, PackageInfo) else PackageInfo(info, parent),
        modules={},
    )


def build_module(
    info: t.Union[str, ModuleInfo],
    parent: t.Optional[PackageInfo] = None,
    resolver: t.Optional[ASTResolver] = None,
    inspector: t.Optional[TypeInspector] = None,
    python_version: t.Union[PythonVersion, t.Sequence[int], None] = None,
) -> ModuleASTBuilder:
    """Start python module builder."""
    return ModuleASTBuilder(
        context=_create_context(resolver, inspector, python_version),
        info=info if isinstance(info, ModuleInfo) else ModuleInfo(info, parent),
        body=[],
    )


def _create_context(
    resolver: t.Optional[ASTResolver],
    inspector: t.Optional[TypeInspector],
    python_version: t.Union[PythonVersion, t.Sequence[int], None],
) -> BuildContext:
    version = PythonVersion.get(python_version)
    inspector = inspector if inspector is not None else TypeInspector()

    return BuildContext(
        version=version,
        packages=[],
        dependencies=defaultdict(set),
        scopes=deque(),
        resolver=resolver if resolver is not None else DefaultASTResolver(inspector, python_version=version),
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

    def _normalize_expr(self, expr: TypeExpr, *tail: str) -> ast.expr:
        return self._context.resolver.resolve_expr(expr, *tail)

    def _normalize_body(self, body: t.Sequence[Stmt], docs: t.Optional[t.Sequence[str]] = None) -> list[ast.stmt]:
        return self._context.resolver.resolve_stmts(*body, docs=docs, pass_if_empty=True)

    def _normalize_type_ref(self, ref: TypeRef) -> TypeInfo:
        return (
            ref
            if isinstance(ref, (ModuleInfo, TypeVarInfo, NamedTypeInfo, LiteralTypeInfo, EnumTypeInfo, UnionTypeInfo))
            else ref.info
            if isinstance(ref, TypeDefinitionBuilder)
            else self._context.inspector.inspect(ref)
        )


# noinspection PyTypeChecker
class BaseASTExpressionBuilder(_BaseBuilder, ASTExpressionBuilder):
    def __init__(self, context: BuildContext, factory: t.Callable[[], Expr]) -> None:
        super().__init__(context)
        self.__factory = factory

    def __neg__(self) -> BaseASTExpressionBuilder:
        return self.__unary_op_expr(ast.Not())

    def __invert__(self) -> BaseASTExpressionBuilder:
        return self.__unary_op_expr(ast.Invert())

    def __and__(self, other: Expr) -> BaseASTExpressionBuilder:
        return self.__bool_op_expr(ast.And(), other)

    def __or__(self, other: Expr) -> BaseASTExpressionBuilder:
        return self.__bool_op_expr(ast.Or(), other)

    def __add__(self, other: Expr) -> BaseASTExpressionBuilder:
        return self.__bin_op_expr(ast.Add(), other)

    def __sub__(self, other: Expr) -> BaseASTExpressionBuilder:
        return self.__bin_op_expr(ast.Sub(), other)

    def __mul__(self, other: Expr) -> BaseASTExpressionBuilder:
        return self.__bin_op_expr(ast.Mult(), other)

    def __matmul__(self, other: Expr) -> BaseASTExpressionBuilder:
        return self.__bin_op_expr(ast.MatMult(), other)

    def __truediv__(self, other: Expr) -> BaseASTExpressionBuilder:
        return self.__bin_op_expr(ast.Div(), other)

    @override
    def build_expr(self) -> ast.expr:
        return self._normalize_expr(self.__factory())

    def stmt(self, *, append: bool = True) -> ast.stmt:
        node = ast.Expr(value=self.build_expr())
        if append:
            self._context.append_body(node)

        return node

    def __bool_op_expr(self, op: ast.boolop, right: Expr) -> BaseASTExpressionBuilder:
        def create() -> ast.expr:
            return ast.BoolOp(op=op, values=[self.build_expr(), self._normalize_expr(right)])

        return BaseASTExpressionBuilder(self._context, create)

    def __unary_op_expr(self, op: ast.unaryop) -> BaseASTExpressionBuilder:
        def create() -> ast.expr:
            return ast.UnaryOp(op=op, operand=self.build_expr())

        return BaseASTExpressionBuilder(self._context, create)

    def __bin_op_expr(self, op: ast.operator, right: Expr) -> BaseASTExpressionBuilder:
        def create() -> ast.expr:
            return ast.BinOp(left=self.build_expr(), op=op, right=self._normalize_expr(right))

        return BaseASTExpressionBuilder(self._context, create)


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


# noinspection PyTypeChecker
class AttrASTBuilder(BaseASTExpressionBuilder):
    def __init__(
        self,
        context: BuildContext,
        head: t.Union[str, TypeExpr],
        *tail: str,
        is_awaited: bool = False,
    ) -> None:
        super().__init__(context, self.__create_expr)
        self.__head = head
        self.__tail = tail
        self.__is_awaited = is_awaited

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

    def await_(self, *, is_awaited: bool = True) -> Self:
        self.__is_awaited = is_awaited
        return self

    def attr(self, *tail: str) -> Self:
        return self.__class__(self._context, self, *tail)

    def call(
        self,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> CallASTBuilder:
        return CallASTBuilder(context=self._context, func=self, args=args, kwargs=kwargs)

    def index(self, index: Expr) -> SliceASTBuilder:
        return SliceASTBuilder(context=self._context, value=self, index=index)

    def slice(
        self,
        lower: t.Optional[Expr] = None,
        upper: t.Optional[Expr] = None,
        step: t.Optional[Expr] = None,
    ) -> SliceASTBuilder:
        return SliceASTBuilder(context=self._context, value=self, index=Slice(lower, upper, step))

    def assign(self, value: Expr) -> ast.stmt:
        return self._scope.assign_stmt(self, value)

    def __create_expr(self) -> ast.expr:
        node: ast.expr = self._normalize_expr(
            ast.Name(id=self.__head) if isinstance(self.__head, str) else self.__head,
            *self.__tail,
        )

        if self.__is_awaited:
            node = ast.Await(value=node)

        return node


# noinspection PyTypeChecker
class CallASTBuilder(BaseASTExpressionBuilder):
    def __init__(
        self,
        context: BuildContext,
        func: TypeExpr,
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

    def index(self, index: Expr) -> SliceASTBuilder:
        return SliceASTBuilder(context=self._context, value=self, index=index)

    def slice(
        self,
        lower: t.Optional[Expr] = None,
        upper: t.Optional[Expr] = None,
        step: t.Optional[Expr] = None,
    ) -> SliceASTBuilder:
        return SliceASTBuilder(context=self._context, value=self, index=Slice(lower, upper, step))

    def __create_expr(self) -> ast.expr:
        node: ast.expr = ast.Call(
            func=self._normalize_expr(self.__func),
            args=[self._normalize_expr(arg) for arg in self.__args],
            keywords=[ast.keyword(arg=key, value=self._normalize_expr(kwarg)) for key, kwarg in self.__kwargs.items()],
            lineno=0,
        )

        if self.__is_awaited:
            node = ast.Await(value=node)

        return node


@dataclass(frozen=True)
class Slice:
    lower: t.Optional[Expr] = None
    upper: t.Optional[Expr] = None
    step: t.Optional[Expr] = None


# noinspection PyTypeChecker
class SliceASTBuilder(BaseASTExpressionBuilder):
    def __init__(
        self,
        context: BuildContext,
        value: TypeExpr,
        index: t.Optional[t.Union[Expr, Slice]] = None,
        *,
        is_awaited: bool = False,
    ) -> None:
        super().__init__(context, self.__create_expr)
        self.__value = value
        self.__index = index
        self.__is_awaited = is_awaited

    def await_(self, *, is_awaited: bool = True) -> Self:
        self.__is_awaited = is_awaited
        return self

    def attr(self, *tail: str) -> AttrASTBuilder:
        return AttrASTBuilder(self._context, self, *tail, is_awaited=self.__is_awaited)

    def call(
        self,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> CallASTBuilder:
        return CallASTBuilder(context=self._context, func=self, args=args, kwargs=kwargs)

    def index(self, index: Expr) -> Self:
        return self.__class__(self._context, self, index)

    def slice(
        self,
        lower: t.Optional[Expr] = None,
        upper: t.Optional[Expr] = None,
        step: t.Optional[Expr] = None,
    ) -> Self:
        return self.__class__(self._context, self, Slice(lower, upper, step))

    def __create_expr(self) -> ast.expr:
        node = self._normalize_expr(self.__value)

        if self.__index is not None:
            node = ast.Subscript(
                value=node,
                slice=ast.Slice(
                    lower=self._normalize_expr(self.__index.lower) if self.__index.lower is not None else None,
                    upper=self._normalize_expr(self.__index.upper) if self.__index.upper is not None else None,
                    step=self._normalize_expr(self.__index.step) if self.__index.step is not None else None,
                )
                if isinstance(self.__index, Slice)
                else self._normalize_expr(self.__index),
            )

        if self.__is_awaited:
            node = ast.Await(value=node)

        return node


class TypeRefBuilder(_BaseBuilder, TypeDefinitionBuilder, ASTExpressionBuilder):
    def __init__(
        self,
        context: BuildContext,
        info: TypeInfo,
        transform: t.Optional[t.Callable[[TypeInfo], TypeInfo]] = None,
    ) -> None:
        super().__init__(context)
        self.__info = info
        self.__transform: t.Callable[[TypeInfo], TypeInfo] = transform if transform is not None else self.__ident

    @override
    @property
    def info(self) -> TypeInfo:
        return self.__transform(self.__info)

    @override
    def ref(self) -> ASTExpressionBuilder:
        return self

    def optional(self) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return predef().optional.with_type_params(inner(info))

        return self.__wrap(transform)

    def union(self, *others: TypeRef) -> TypeRefBuilder:
        if not others:
            return self

        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return UnionTypeInfo(values=(inner(info), *(self._normalize_type_ref(type_) for type_ in others)))

        return self.__wrap(transform)

    def collection(self) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return predef().collection.with_type_params(inner(info))

        return self.__wrap(transform)

    def set(self) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return predef().set.with_type_params(inner(info))

        return self.__wrap(transform)

    def sequence(self, *, mutable: bool = False) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return (predef().mutable_sequence if mutable else predef().sequence).with_type_params(inner(info))

        return self.__wrap(transform)

    def list(self) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return predef().list.with_type_params(inner(info))

        return self.__wrap(transform)

    def mapping_key(self, value: TypeRef, *, mutable: bool = False) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return (predef().mutable_mapping if mutable else predef().mapping).with_type_params(
                inner(info), self._normalize_type_ref(value)
            )

        return self.__wrap(transform)

    def dict_key(self, value: TypeRef) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return predef().dict.with_type_params(inner(info), self._normalize_type_ref(value))

        return self.__wrap(transform)

    def mapping_value(self, key: TypeRef, *, mutable: bool = False) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return (predef().mutable_mapping if mutable else predef().mapping).with_type_params(
                self._normalize_type_ref(key), inner(info)
            )

        return self.__wrap(transform)

    def dict_value(self, key: TypeRef) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return predef().dict.with_type_params(self._normalize_type_ref(key), inner(info))

        return self.__wrap(transform)

    def context_manager(self, *, is_async: bool = False) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return (predef().async_context_manager if is_async else predef().context_manager).with_type_params(
                inner(info)
            )

        return self.__wrap(transform)

    def iterator(self, *, is_async: bool = False) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return (predef().async_iterator if is_async else predef().iterator).with_type_params(inner(info))

        return self.__wrap(transform)

    def iterable(self, *, is_async: bool = False) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            return (predef().async_iterable if is_async else predef().iterable).with_type_params(inner(info))

        return self.__wrap(transform)

    def type_params(self, *params: TypeRef) -> TypeRefBuilder:
        def transform(
            inner: t.Callable[[TypeInfo], TypeInfo],
            info: TypeInfo,
        ) -> TypeInfo:
            origin = inner(info)

            if not isinstance(origin, NamedTypeInfo):
                msg = "named type info was expected to apply type params"
                raise TypeError(msg, origin, params)

            return origin.with_type_params(*(self._normalize_type_ref(param) for param in params))

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
        return self._context.resolver.resolve_expr(self.info)

    def __ident(self, info: TypeInfo) -> TypeInfo:
        return info

    def __wrap(
        self,
        transform: t.Callable[[t.Callable[[TypeInfo], TypeInfo], TypeInfo], TypeInfo],
    ) -> TypeRefBuilder:
        return self.__class__(self._context, self.__info, partial(transform, self.__transform))


class AnnotationASTBuilder(_BaseBuilder):
    def type_ref(self, origin: TypeRef) -> TypeRefBuilder:
        return TypeRefBuilder(self._context, self._normalize_type_ref(origin))

    if sys.version_info >= (3, 10):

        @_ast_expr_builder
        def const(self, value: t.Union[str, bytes, bool, complex, EllipsisType, None]) -> Expr:  # noqa: FBT001
            return ast.Constant(value=value)

    else:

        @_ast_expr_builder
        def const(self, value: t.Union[str, bytes, bool, complex, None]) -> Expr:  # noqa: FBT001
            return ast.Constant(value=value)

    def none(self) -> Expr:
        return self.const(None)

    def ellipsis(self) -> Expr:
        return ast.Constant(value=...)

    def generic_type(self, generic: TypeExpr, *params: TypeExpr) -> Expr:
        if len(params) == 0:
            return self._normalize_expr(generic)

        return ast.Subscript(
            value=self._normalize_expr(generic),
            slice=self._normalize_expr(self.tuple_type(*params, normalize=True)),
        )

    def literal_type(self, *values: t.Union[str, Expr]) -> Expr:
        if not values:
            return self._normalize_expr(predef().no_return)

        return self.generic_type(
            predef().literal,
            *(self.const(val) if isinstance(val, str) else val for val in values),
        )

    def optional_type(self, of_type: TypeExpr) -> Expr:
        return self.generic_type(predef().optional, of_type)

    def union_type(self, *params: TypeExpr, normalize: bool = False) -> Expr:
        if not params:
            return self._normalize_expr(predef().no_return)

        if normalize and len(params) == 1:
            return self._normalize_expr(params[0])

        return self.generic_type(predef().union, *params)

    def tuple_type(self, *params: TypeExpr, normalize: bool = False) -> Expr:
        if normalize and len(params) == 1:
            return self._normalize_expr(params[0])

        return ast.Tuple(elts=[self._normalize_expr(item) for item in params])

    def collection_type(self, of_type: TypeExpr) -> Expr:
        return self.generic_type(predef().collection, of_type)

    def sequence_type(self, of_type: TypeExpr, *, mutable: bool = False) -> Expr:
        return self.generic_type(predef().mutable_sequence if mutable else predef().sequence, of_type)

    def list_type(self, of_type: TypeExpr) -> Expr:
        return self.generic_type(predef().list, of_type)

    def mapping_type(self, key_type: TypeExpr, value_type: TypeExpr, *, mutable: bool = False) -> Expr:
        return self.generic_type(predef().mutable_mapping if mutable else predef().mapping, key_type, value_type)

    def dict_type(self, key_type: TypeExpr, value_type: TypeExpr) -> Expr:
        return self.generic_type(predef().dict, key_type, value_type)

    def iterator_type(self, of_type: TypeExpr, *, is_async: bool = False) -> Expr:
        return self.generic_type(predef().async_iterator if is_async else predef().iterator, of_type)

    def iterable_type(self, of_type: TypeExpr, *, is_async: bool = False) -> Expr:
        return self.generic_type(predef().async_iterable if is_async else predef().iterable, of_type)

    def context_manager_type(self, of_type: TypeExpr, *, is_async: bool = False) -> Expr:
        return self.generic_type(
            predef().async_context_manager if is_async else predef().context_manager,
            of_type,
        )


@dataclass(frozen=True)
class Comprehension:
    target: Expr
    items: Expr
    predicates: t.Sequence[Expr] = field(default_factory=list)
    is_async: bool = False


# noinspection PyTypeChecker
class ScopeASTBuilder(AnnotationASTBuilder):
    @_ast_expr_builder
    def await_(self, expr: Expr) -> Expr:
        return ast.Await(self._normalize_expr(expr))

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
    def tuple_expr(self, *items: TypeExpr, normalize: bool = False) -> Expr:
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

    @_ast_expr_builder
    def compare_in_expr(self, left: Expr, right: Expr) -> Expr:
        return ast.Compare(
            left=self._normalize_expr(left),
            ops=[ast.In()],
            comparators=[self._normalize_expr(right)],
        )

    @_ast_expr_builder
    def compare_not_in_expr(self, left: Expr, right: Expr) -> Expr:
        return ast.Compare(
            left=self._normalize_expr(left),
            ops=[ast.NotIn()],
            comparators=[self._normalize_expr(right)],
        )

    def attr(self, head: t.Union[str, TypeExpr], *tail: str) -> AttrASTBuilder:
        return AttrASTBuilder(self._context, head, *tail)

    def call(
        self,
        func: TypeExpr,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> CallASTBuilder:
        return CallASTBuilder(self._context, func, args, kwargs)

    @_ast_expr_builder
    def subscript(self, value: TypeExpr, *slice_: TypeExpr) -> Expr:
        return ast.Subscript(
            value=self._normalize_expr(value),
            slice=self._normalize_expr(self.tuple_expr(*slice_, normalize=True)),
        )

    @_ast_expr_builder
    def slice(
        self,
        lower: t.Optional[Expr] = None,
        upper: t.Optional[Expr] = None,
        step: t.Optional[Expr] = None,
    ) -> Expr:
        return ast.Slice(
            lower=self._normalize_expr(lower) if lower is not None else None,
            upper=self._normalize_expr(upper) if upper is not None else None,
            step=self._normalize_expr(step) if step is not None else None,
        )

    def stmt(self, *stmts: t.Optional[Stmt]) -> None:
        self._context.extend_body(self._context.resolver.resolve_stmts(*stmts))

    def compr(
        self,
        target: Expr,
        items: Expr,
        predicates: t.Optional[t.Sequence[Expr]] = None,
        *,
        is_async: bool = False,
    ) -> Comprehension:
        return Comprehension(target=target, items=items, predicates=predicates or (), is_async=is_async)

    def class_def(self, name: str) -> ClassStatementASTBuilder:
        return ClassStatementASTBuilder(self._context, name)

    def func_def(self, name: str) -> FuncStatementASTBuilder:
        return FuncStatementASTBuilder(self._context, name)

    @_ast_stmt_builder
    def field_def(self, name: str, annotation: TypeExpr, default: t.Optional[Expr] = None) -> ast.stmt:
        return ast.AnnAssign(
            target=ast.Name(id=name),
            annotation=self._normalize_expr(annotation),
            value=self._normalize_expr(default) if default is not None else None,
            simple=1,
        )

    def type_alias(self, name: str) -> TypeAliasStatementASTBuilder:
        return TypeAliasStatementASTBuilder(self._context, name)

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
            raise ASTBuildError(msg, self, body)

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
        types: t.Sequence[TypeExpr]
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

    def except_(self, *types: TypeExpr, name: t.Optional[str] = None) -> t.ContextManager[ScopeASTBuilder]:
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


class TypeVarBuilder(_BaseBuilder, TypeDefinitionBuilder, ASTStatementBuilder):
    def __init__(
        self,
        context: BuildContext,
        name: str,
        variance: t.Optional[TypeVarVariance] = None,
        constraints: t.Optional[t.Sequence[TypeExpr]] = None,
        lower: t.Optional[TypeExpr] = None,
    ) -> None:
        super().__init__(context)
        self.__name = name
        self.__module = self._context.module
        self.__namespace = (
            self._context.namespace if context.version >= PythonVersion.PY312 else self._context.namespace[:-1]
        )
        self.__variance = variance
        self.__constraints = list[TypeExpr](constraints or ())
        self.__lower = lower

        self.__info = NamedTypeInfo(
            name=self.__name,
            module=self.__module,
            namespace=self.__namespace,
        )

    def __enter__(self) -> TypeInfo:
        return self.__info

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass

    @override
    @property
    def info(self) -> TypeInfo:
        return self.__info

    @override
    def ref(self) -> ASTExpressionBuilder:
        return TypeRefBuilder(self._context, self.__info)

    def invariant(self) -> Self:
        self.__variance = "invariant"
        return self

    def covariant(self) -> Self:
        self.__variance = "covariant"
        return self

    def contravariant(self) -> Self:
        self.__variance = "contravariant"
        return self

    def constraints(self, *types: TypeExpr) -> Self:
        self.__constraints.extend(types)
        return self

    def lower(self, type_: TypeExpr) -> Self:
        self.__lower = type_
        return self

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        args: list[ast.expr] = [ast.Constant(value=self.__name)]
        keywords = list[ast.keyword]()

        if self.__variance is None or self.__variance == "invariant":
            pass
        elif self.__variance == "covariant":
            keywords.append(ast.keyword(arg="covariant", value=ast.Constant(value=True)))
        elif self.__variance == "contravariant":
            keywords.append(ast.keyword(arg="contravariant", value=ast.Constant(value=True)))
        else:
            assert_never(self.__variance)

        if self.__lower is not None:
            keywords.append(ast.keyword(arg="bound", value=self._normalize_expr(self.__lower)))

        return [
            ast.Assign(
                targets=[ast.Name(id=self.__name)],
                value=ast.Call(
                    func=self._normalize_expr(predef().type_var),
                    args=args,
                    keywords=keywords,
                ),
                lineno=0,
            ),
        ]

    # NOTE: workaround for passing mypy typings in CI for python 3.12
    if sys.version_info >= (3, 12):

        def build_type_param(self) -> ast.type_param:
            return ast.TypeVar(
                name=self.__name,
                bound=self._normalize_expr(self.__lower) if self.__lower is not None else None,
            )


class TypeAliasExpressionBuilder(AnnotationASTBuilder, TypeDefinitionBuilder, ASTStatementBuilder):
    def __init__(self, context: BuildContext, info: NamedTypeInfo) -> None:
        super().__init__(context)
        self.__info = info
        self.__expr: t.Optional[TypeExpr] = None
        self.__type_vars = list[TypeVarBuilder]()

    @override
    @property
    def info(self) -> TypeInfo:
        return self.__info

    @override
    def ref(self) -> TypeRefBuilder:
        return TypeRefBuilder(self._context, self.__info)

    def assign(self, expr: TypeExpr) -> None:
        self.__expr = expr

    def type_var(self, name: str) -> TypeVarBuilder:
        type_var = TypeVarBuilder(self._context, name)
        self.__type_vars.append(type_var)
        return type_var

    def type_params(self, *params: TypeExpr) -> Expr:
        return self.ref().type_params(*params)

    # NOTE: workaround for passing mypy typings in CI for python 3.12
    if sys.version_info >= (3, 12):

        @override
        def build_stmt(self) -> t.Sequence[ast.stmt]:
            if self.__expr is None:
                msg = "type alias expression is not set"
                raise IncompleteStatementError(msg, self)

            return (
                [
                    ast.TypeAlias(
                        name=ast.Name(id=self.__info.name),
                        value=self._normalize_expr(self.__expr),
                        type_params=[tv.build_type_param() for tv in self.__type_vars],
                    )
                ]
                if self._context.version >= PythonVersion.PY312
                else self.__build_ann_assign()
            )

    else:

        @override
        def build_stmt(self) -> t.Sequence[ast.stmt]:
            return self.__build_ann_assign()

    def __build_ann_assign(self) -> t.Sequence[ast.stmt]:
        if self.__expr is None:
            msg = "type alias expression is not set"
            raise IncompleteStatementError(msg, self)

        stmts = [stmt for tv in self.__type_vars for stmt in tv.build_stmt()]
        stmts.append(
            ast.AnnAssign(
                target=ast.Name(id=self.__info.name),
                annotation=self._normalize_expr(predef().type_alias),
                value=self._normalize_expr(self.__expr),
                simple=1,
            )
        )

        return stmts


class TypeAliasStatementASTBuilder(_BaseBuilder, ASTStatementBuilder, TypeDefinitionBuilder):
    def __init__(self, context: BuildContext, name: str) -> None:
        super().__init__(context)
        self.__info = NamedTypeInfo(name=name, module=self._context.module, namespace=self._context.namespace)
        self.__annotation = TypeAliasExpressionBuilder(context=self._context, info=self.__info)

    def __enter__(self) -> TypeAliasExpressionBuilder:
        self._context.enter_scope(self.__annotation.info.name, [])
        return self.__annotation

    def __exit__(self, exc_type: object, exc_value: object, exc_traceback: object) -> None:
        if exc_type is None:
            self._context.leave_scope()
            self._context.extend_body(self.build_stmt())

    @override
    @property
    def info(self) -> TypeInfo:
        return self.__info

    @override
    def ref(self) -> ASTExpressionBuilder:
        return TypeRefBuilder(self._context, self.__info)

    def assign(self, expr: TypeExpr) -> None:
        self.__annotation.assign(expr)
        self._context.extend_body(self.build_stmt())

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        return self.__annotation.build_stmt()


class ClassScopeASTBuilder(ScopeASTBuilder, TypeDefinitionBuilder):
    def __init__(self, context: BuildContext, header: ClassStatementASTBuilder) -> None:
        super().__init__(context)
        self.__header = header

    @override
    @property
    def info(self) -> TypeInfo:
        return self.__header.info

    @override
    def ref(self) -> TypeRefBuilder:
        return self.__header.ref()

    def type_var(self, name: str) -> TypeVarBuilder:
        return self.__header.type_var(name)

    def method_def(self, name: str) -> MethodStatementASTBuilder:
        return MethodStatementASTBuilder(self._context, name)

    def new_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__new__")

    def init_def(self) -> MethodStatementASTBuilder:
        return self.method_def("__init__").returns(self.const(None))

    @contextmanager
    def init_self_attrs_def(self, attrs: t.Mapping[str, TypeExpr]) -> t.Iterator[MethodScopeASTBuilder]:
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
        self.__bases = list[TypeExpr]()
        self.__decorators = list[TypeExpr]()
        self.__keywords = dict[str, TypeExpr]()
        self.__type_vars = list[TypeVarBuilder]()
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
    def ref(self) -> TypeRefBuilder:
        return TypeRefBuilder(self._context, self.__info)

    def type_var(self, name: str) -> TypeVarBuilder:
        type_var = TypeVarBuilder(self._context, name)
        self.__type_vars.append(type_var)
        return type_var

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

        if kw_only and self._context.version >= PythonVersion.PY310:
            dc.kwarg("kw_only", ast.Constant(value=kw_only))

        return self.decorators(dc)

    def inherits(self, *bases: t.Optional[TypeExpr]) -> Self:
        self.__bases.extend(base for base in bases if base is not None)
        return self

    def decorators(self, *items: t.Optional[TypeExpr]) -> Self:
        self.__decorators.extend(item for item in items if item is not None)
        return self

    def keywords(self, **keywords: t.Optional[TypeExpr]) -> Self:
        self.__keywords.update({key: value for key, value in keywords.items() if value is not None})
        return self

    # NOTE: workaround for passing mypy typings in CI for python 3.12
    if sys.version_info >= (3, 12):
        # noinspection PyArgumentList
        @override
        def build_stmt(self) -> t.Sequence[ast.stmt]:
            return (
                [
                    ast.ClassDef(
                        name=self.__info.name,
                        bases=self.__build_bases(),
                        keywords=self.__build_keywords(),
                        body=self._normalize_body(self.__body, self.__docs),
                        decorator_list=self.__build_decorators(),
                        type_params=[tv.build_type_param() for tv in self.__type_vars],
                    ),
                ]
                if self._context.version >= PythonVersion.PY312
                else self.__build_type_vars_and_class()
            )

    else:
        # noinspection PyArgumentList
        @override
        def build_stmt(self) -> t.Sequence[ast.stmt]:
            return self.__build_type_vars_and_class()

    def __build_type_vars_and_class(self) -> t.Sequence[ast.stmt]:
        stmts = [stmt for tv in self.__type_vars for stmt in tv.build_stmt()]

        if self.__type_vars:
            self.__bases.insert(0, predef().generic.with_type_params(*(tv.info for tv in self.__type_vars)))

        stmts.append(self.__build_class())

        return stmts

    if sys.version_info >= (3, 12):

        def __build_class(self) -> ast.ClassDef:
            return ast.ClassDef(
                name=self.__info.name,
                bases=self.__build_bases(),
                keywords=self.__build_keywords(),
                body=self._normalize_body(self.__body, self.__docs),
                decorator_list=self.__build_decorators(),
                type_params=[],
            )
    else:

        def __build_class(self) -> ast.ClassDef:
            return ast.ClassDef(
                name=self.__info.name,
                bases=self.__build_bases(),
                keywords=self.__build_keywords(),
                body=self._normalize_body(self.__body, self.__docs),
                decorator_list=self.__build_decorators(),
            )

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


@dataclass(frozen=True)
class FuncArgInfo:
    name: str
    kind: t.Literal["positional-only", "positional-or-keyword", "var-positional", "keyword-only", "var-keyword"]
    annotation: t.Optional[TypeExpr] = None
    default: t.Optional[Expr] = None


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
        self.__decorators = list[TypeExpr]()
        self.__args = list[FuncArgInfo]()
        self.__returns: t.Optional[TypeExpr] = None
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

    def decorators(self, *items: t.Optional[TypeExpr]) -> Self:
        self.__decorators.extend(item for item in items if item is not None)
        return self

    def arg(
        self,
        name: str,
        annotation: t.Optional[TypeExpr] = None,
        default: t.Optional[Expr] = None,
    ) -> Self:
        return self.args(FuncArgInfo(name=name, kind="positional-or-keyword", annotation=annotation, default=default))

    def kwarg(
        self,
        name: str,
        annotation: t.Optional[TypeExpr] = None,
        default: t.Optional[Expr] = None,
    ) -> Self:
        return self.args(FuncArgInfo(name=name, kind="keyword-only", annotation=annotation, default=default))

    @t.overload
    def args(self, *args: FuncArgInfo) -> Self: ...

    @t.overload
    def args(self, *args: t.Sequence[FuncArgInfo]) -> Self: ...

    def args(self, *args: t.Union[FuncArgInfo, t.Sequence[FuncArgInfo]]) -> Self:
        self.__args.extend(chain.from_iterable((part,) if isinstance(part, FuncArgInfo) else part for part in args))
        return self

    def returns(self, ret: t.Optional[TypeExpr]) -> Self:
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

        if self.__is_async:
            # noinspection PyArgumentList
            node = ast.AsyncFunctionDef(  # type: ignore[call-overload,no-any-return,unused-ignore]
                # type_comment and type_params has default value each in 3.12 and not available in 3.9
                name=self.__info.name,
                decorator_list=self.__build_decorators(),
                args=self.__build_args(),
                returns=self.__build_returns(),
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
                returns=self.__build_returns(),
                body=self.__build_body(),
                lineno=0,
            )

        return [node]

    def __build_decorators(self) -> list[ast.expr]:
        head_decorators: list[TypeExpr] = []
        last_decorators: list[TypeExpr] = []

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
        node = ast.arguments(
            posonlyargs=[],
            args=[],
            defaults=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
        )

        for info in self.__args:
            arg = ast.arg(
                arg=info.name,
                annotation=self._normalize_expr(info.annotation) if info.annotation is not None else None,
            )

            if info.kind == "positional-only":
                node.posonlyargs.append(arg)

            elif info.kind == "positional-or-keyword":
                node.args.append(arg)
                if info.default is not None:
                    node.defaults.append(self._normalize_expr(info.default))

            elif info.kind == "var-positional":
                node.vararg = arg

            elif info.kind == "keyword-only":
                node.kwonlyargs.append(arg)
                node.kw_defaults.append(self._normalize_expr(info.default) if info.default is not None else None)

            elif info.kind == "var-keyword":
                node.kwarg = arg

            else:
                assert_never(info.kind)

        return node

    def __build_returns(self) -> t.Optional[ast.expr]:
        if self.__returns is None:
            return None

        ret = self.__returns
        if self.__iterator_cm:
            ret = ast.Subscript(
                value=self._normalize_expr(predef().async_iterator if self.__is_async else predef().iterator),
                slice=self._normalize_expr(ret),
            )

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
