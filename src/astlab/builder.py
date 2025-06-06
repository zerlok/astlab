from __future__ import annotations

__all__ = [
    "AttrASTBuilder",
    "CallASTBuilder",
    "ClassBodyASTBuilder",
    "ClassHeaderASTBuilder",
    "ClassRefBuilder",
    "ForHeaderASTBuilder",
    "FuncBodyASTBuilder",
    "FuncHeaderASTBuilder",
    "MethodBodyASTBuilder",
    "MethodHeaderASTBuilder",
    "ModuleASTBuilder",
    "PackageASTBuilder",
    "ScopeASTBuilder",
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
from dataclasses import dataclass
from functools import cached_property
from itertools import chain

from astlab._typing import Self, override
from astlab.abc import (
    ASTExpressionBuilder,
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


class AttrASTBuilder(ASTExpressionBuilder):
    def __init__(self, context: BuildContext, head: t.Union[str, TypeRef], *tail: str) -> None:
        self.__context = context
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
        return self.__class__(self.__context, self, *tail)

    def call(
        self,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> CallASTBuilder:
        return CallASTBuilder(
            context=self.__context,
            func=self,
            args=args,
            kwargs=kwargs,
        )

    @override
    def build_expr(self) -> ast.expr:
        return self.__context.resolver.resolve_expr(
            ast.Name(id=self.__head) if isinstance(self.__head, str) else self.__head,
            *self.__tail,
        )


# noinspection PyTypeChecker
class CallASTBuilder(ASTExpressionBuilder):
    def __init__(
        self,
        context: BuildContext,
        func: TypeRef,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> None:
        self.__context = context
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
        self.__args.append(self.__context.resolver.resolve_expr(expr))
        return self

    def kwarg(self, name: str, expr: Expr) -> Self:
        self.__kwargs[name] = expr
        return self

    def attr(self, *tail: str) -> AttrASTBuilder:
        return AttrASTBuilder(self.__context, self).attr(*tail)

    def call(
        self,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> Self:
        return self.__class__(
            context=self.__context,
            func=self,
            args=args,
            kwargs=kwargs,
        )

    @override
    def build_expr(self) -> ast.expr:
        node: ast.expr = ast.Call(
            func=self.__context.resolver.resolve_expr(self.__func),
            args=[self.__context.resolver.resolve_expr(arg) for arg in self.__args],
            keywords=[
                ast.keyword(arg=key, value=self.__context.resolver.resolve_expr(kwarg))
                for key, kwarg in self.__kwargs.items()
            ],
            lineno=0,
        )

        if self.__is_awaited:
            node = ast.Await(value=node)

        return node


# noinspection PyTypeChecker
class _BaseASTBuilder:
    def __init__(self, context: BuildContext) -> None:
        self._context = context

    def const(self, value: object) -> ast.expr:
        assert not isinstance(value, ast.AST)
        return ast.Constant(value=value)

    def none(self) -> ast.expr:
        return self.const(None)

    def ternary_not_none_expr(
        self,
        body: Expr,
        test: Expr,
        or_else: t.Optional[Expr] = None,
    ) -> ast.expr:
        return ast.IfExp(
            test=ast.Compare(
                left=self._context.resolver.resolve_expr(test), ops=[ast.IsNot()], comparators=[self.none()]
            ),
            body=self._context.resolver.resolve_expr(body),
            orelse=self._context.resolver.resolve_expr(or_else) if or_else is not None else self.none(),
        )

    def tuple_expr(self, *items: TypeRef) -> ast.expr:
        return ast.Tuple(elts=[self._context.resolver.resolve_expr(item) for item in items])

    @t.overload
    def set_expr(self, items: Expr, target: Expr, item: Expr) -> ast.expr: ...

    @t.overload
    def set_expr(self, items: t.Collection[Expr]) -> ast.expr: ...

    def set_expr(
        self,
        items: t.Union[Expr, t.Collection[Expr]],
        target: t.Optional[Expr] = None,
        item: t.Optional[Expr] = None,
    ) -> ast.expr:
        if isinstance(items, (ast.expr, ASTExpressionBuilder)):
            assert target is not None
            assert item is not None

            return ast.SetComp(
                elt=self._context.resolver.resolve_expr(item),
                generators=[
                    ast.comprehension(
                        target=self._context.resolver.resolve_expr(target),
                        iter=self._context.resolver.resolve_expr(items),
                        ifs=[],
                        is_async=False,
                    )
                ],
            )

        return ast.Set(elts=[self._context.resolver.resolve_expr(item) for item in items])

    @t.overload
    def list_expr(self, items: Expr, target: Expr, item: Expr) -> ast.expr: ...

    @t.overload
    def list_expr(self, items: t.Sequence[Expr]) -> ast.expr: ...

    def list_expr(
        self,
        items: t.Union[Expr, t.Sequence[Expr]],
        target: t.Optional[Expr] = None,
        item: t.Optional[Expr] = None,
    ) -> ast.expr:
        if isinstance(items, (ast.expr, ASTExpressionBuilder)):
            assert target is not None
            assert item is not None

            return ast.ListComp(
                elt=self._context.resolver.resolve_expr(item),
                generators=[
                    ast.comprehension(
                        target=self._context.resolver.resolve_expr(target),
                        iter=self._context.resolver.resolve_expr(items),
                        ifs=[],
                        is_async=False,
                    )
                ],
            )

        return ast.List(elts=[self._context.resolver.resolve_expr(item) for item in items])

    @t.overload
    def dict_expr(self, items: Expr, target: Expr, key: Expr, value: Expr) -> ast.expr: ...

    @t.overload
    def dict_expr(self, items: t.Mapping[Expr, Expr]) -> ast.expr: ...

    def dict_expr(
        self,
        items: t.Union[Expr, t.Mapping[Expr, Expr]],
        target: t.Optional[Expr] = None,
        key: t.Optional[Expr] = None,
        value: t.Optional[Expr] = None,
    ) -> ast.expr:
        if isinstance(items, (ast.expr, ASTExpressionBuilder)):
            assert target is not None
            assert key is not None
            assert value is not None

            return ast.DictComp(
                key=self._context.resolver.resolve_expr(key),
                value=self._context.resolver.resolve_expr(value),
                generators=[
                    ast.comprehension(
                        target=self._context.resolver.resolve_expr(target),
                        iter=self._context.resolver.resolve_expr(items),
                        ifs=[],
                        is_async=False,
                    )
                ],
            )

        return ast.Dict(
            keys=[self._context.resolver.resolve_expr(key) for key in items],
            values=[self._context.resolver.resolve_expr(value) for value in items.values()],
        )

    def not_(self, expr: Expr) -> ast.expr:
        return ast.UnaryOp(op=ast.Not(), operand=self._context.resolver.resolve_expr(expr))

    def attr(self, head: t.Union[str, TypeRef], *tail: str) -> AttrASTBuilder:
        return AttrASTBuilder(self._context, head, *tail)

    def call(
        self,
        func: TypeRef,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> CallASTBuilder:
        return self.attr(func).call(args, kwargs)

    def type_ref(self, origin: t.Union[RuntimeType, TypeInfo]) -> ClassRefBuilder:
        return ClassRefBuilder(
            context=self._context,
            info=origin
            if isinstance(origin, (NamedTypeInfo, LiteralTypeInfo))
            else self._context.inspector.inspect(origin),
        )

    def generic_type(self, generic: TypeRef, *args: TypeRef) -> ast.expr:
        if len(args) == 0:
            return self._context.resolver.resolve_expr(generic)

        if len(args) == 1:
            return ast.Subscript(
                value=self._context.resolver.resolve_expr(generic), slice=self._context.resolver.resolve_expr(args[0])
            )

        return ast.Subscript(value=self._context.resolver.resolve_expr(generic), slice=self.tuple_expr(*args))

    def literal_type(self, *args: t.Union[str, Expr]) -> ast.expr:
        if not args:
            return self._context.resolver.resolve_expr(predef().no_return)

        return self.generic_type(
            predef().literal,
            *(self.const(arg) if isinstance(arg, str) else arg for arg in args),
        )

    def optional_type(self, of_type: TypeRef) -> ast.expr:
        return self.generic_type(predef().optional, of_type)

    def sequence_type(self, of_type: TypeRef, *, mutable: bool = False) -> ast.expr:
        return self.generic_type(predef().mutable_sequence if mutable else predef().sequence, of_type)

    def iterator_type(self, of_type: TypeRef, *, is_async: bool = False) -> ast.expr:
        return self.generic_type(predef().async_iterator if is_async else predef().iterator, of_type)

    def iterable_type(self, of_type: TypeRef, *, is_async: bool = False) -> ast.expr:
        return self.generic_type(predef().async_iterable if is_async else predef().iterable, of_type)

    def context_manager_type(self, of_type: TypeRef, *, is_async: bool = False) -> ast.expr:
        return self.generic_type(
            predef().async_context_manager if is_async else predef().context_manager,
            of_type,
        )

    def ellipsis_stmt(self) -> ast.stmt:
        return ast.Expr(value=ast.Constant(value=...))

    def pass_stmt(self) -> ast.stmt:
        return ast.Pass()


class _BaseHeaderASTBuilder(t.Generic[T_co], t.ContextManager[T_co], ASTStatementBuilder, metaclass=abc.ABCMeta):
    def __init__(self, context: BuildContext, name: t.Optional[str] = None) -> None:
        self._context = context
        self._name = name

    @override
    def __enter__(self) -> T_co:
        self._context.enter_scope(self._name, [])
        return self._create_scope_builder()

    @override
    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        exc_traceback: object,
    ) -> None:
        if exc_type is None:
            # 1. build statements using nested context
            body = self._context.resolver.resolve_stmts(self)
            # 2. exit nested context
            self._context.leave_scope()
            # 3. fill statements to current scope
            self._context.extend_body(body)

    @abc.abstractmethod
    def _create_scope_builder(self) -> T_co:
        raise NotImplementedError


# noinspection PyTypeChecker
class ScopeASTBuilder(_BaseASTBuilder):
    def class_def(self, name: str) -> ClassHeaderASTBuilder:
        return ClassHeaderASTBuilder(self._context, name)

    def func_def(self, name: str) -> FuncHeaderASTBuilder:
        return FuncHeaderASTBuilder(self._context, name)

    def field_def(self, name: str, annotation: TypeRef, default: t.Optional[Expr] = None) -> ast.stmt:
        node = ast.AnnAssign(
            target=ast.Name(id=name),
            annotation=self._context.resolver.resolve_expr(annotation),
            value=self._context.resolver.resolve_expr(default) if default is not None else None,
            simple=1,
        )
        self._context.append_body(node)

        return node

    def stmt(self, *stmts: t.Optional[Stmt]) -> None:
        self._context.extend_body(self._context.resolver.resolve_stmts(*stmts))

    def assign_stmt(self, target: t.Union[str, Expr], value: Expr) -> ast.stmt:
        node = ast.Assign(
            targets=[self._context.resolver.resolve_expr(self.attr(target))],
            value=self._context.resolver.resolve_expr(value),
            lineno=0,
        )
        self._context.append_body(node)

        return node

    def if_stmt(self, test: Expr) -> IfStatementASTBuilder:
        return IfStatementASTBuilder(self._context, test)

    def for_stmt(self, target: str, items: Expr) -> ForHeaderASTBuilder:
        return ForHeaderASTBuilder(self._context, target, items)

    def while_stmt(self, test: Expr) -> WhileHeaderASTBuilder:
        return WhileHeaderASTBuilder(self._context, test)

    def break_stmt(self) -> ast.stmt:
        node = ast.Break()
        self._context.append_body(node)
        return node

    def return_stmt(self, value: Expr) -> ast.stmt:
        node = ast.Return(
            value=self._context.resolver.resolve_expr(value),
            lineno=0,
        )
        self._context.append_body(node)

        return node

    def yield_stmt(self, value: Expr) -> ast.stmt:
        node = ast.Expr(
            value=ast.Yield(
                value=self._context.resolver.resolve_expr(value),
                lineno=0,
            ),
        )
        self._context.append_body(node)

        return node

    def try_stmt(self) -> TryStatementASTBuilder:
        return TryStatementASTBuilder(self._context)

    def raise_stmt(self, err: Expr, cause: t.Optional[Expr] = None) -> ast.stmt:
        node = ast.Raise(
            exc=self._context.resolver.resolve_expr(err),
            cause=self._context.resolver.resolve_expr(cause) if cause is not None else None,
        )
        self._context.append_body(node)
        return node

    def with_stmt(self) -> WithStatementASTBuilder:
        return WithStatementASTBuilder(self._context)


class _BaseScopeBodyASTBuilder(ScopeASTBuilder, ASTStatementBuilder):
    def __init__(self, context: BuildContext, parent: ASTStatementBuilder) -> None:
        super().__init__(context)
        self.__parent = parent

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        return self.__parent.build_stmt()


class _BaseChainPartASTBuilder(t.ContextManager[ScopeASTBuilder], ASTStatementBuilder):
    def __init__(self, context: BuildContext, name: t.Optional[str] = None) -> None:
        self._context = context
        self._name = name
        self.__stmts = list[ast.stmt]()

    @override
    def __enter__(self) -> ScopeASTBuilder:
        self._context.enter_scope(self._name, [])
        return _BaseScopeBodyASTBuilder(self._context, self)

    @override
    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        exc_traceback: object,
    ) -> None:
        if exc_type is None:
            # 1. build statements using nested context
            stmts = self._context.resolver.resolve_stmts(*self._context.current_body)
            # 2. exit nested context
            self._context.leave_scope()
            # 3. fill statements to current chain
            self.__stmts.extend(stmts)

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        return self.__stmts


# noinspection PyTypeChecker
class WhileHeaderASTBuilder(_BaseHeaderASTBuilder[ScopeASTBuilder]):
    def __init__(self, context: BuildContext, test: Expr) -> None:
        super().__init__(context)
        self.__test = test

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        return [
            ast.While(
                test=self._context.resolver.resolve_expr(self.__test),
                body=self._context.current_body,
                orelse=[],
                lineno=0,
            ),
        ]

    @override
    def _create_scope_builder(self) -> _BaseScopeBodyASTBuilder:
        return _BaseScopeBodyASTBuilder(self._context, self)


# noinspection PyTypeChecker
class ForHeaderASTBuilder(_BaseHeaderASTBuilder[ScopeASTBuilder]):
    def __init__(self, context: BuildContext, target: str, items: Expr) -> None:
        super().__init__(context)
        self.__target = target
        self.__items = items
        self.__is_async = False

    def async_(self, *, is_async: bool = True) -> Self:
        self.__is_async = is_async
        return self

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        stmt = (
            ast.AsyncFor(
                target=ast.Name(id=self.__target),
                iter=self._context.resolver.resolve_expr(self.__items),
                body=self._context.current_body,
                orelse=[],
                lineno=0,
            )
            if self.__is_async
            else ast.For(
                target=ast.Name(id=self.__target),
                iter=self._context.resolver.resolve_expr(self.__items),
                body=self._context.current_body,
                orelse=[],
                lineno=0,
            )
        )

        return [stmt]

    @override
    def _create_scope_builder(self) -> _BaseScopeBodyASTBuilder:
        return _BaseScopeBodyASTBuilder(self._context, self)


# noinspection PyTypeChecker
class WithStatementASTBuilder(_BaseHeaderASTBuilder[ScopeASTBuilder]):
    def __init__(self, context: BuildContext) -> None:
        super().__init__(context)
        self.__cms = list[Expr]()
        self.__names = list[t.Optional[str]]()
        self.__is_async = False

    def async_(self, *, is_async: bool = True) -> Self:
        self.__is_async = is_async
        return self

    def enter(self, cm: Expr, name: t.Optional[str] = None) -> Self:
        self.__cms.append(cm)
        self.__names.append(name)
        return self

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        items = [
            ast.withitem(
                self._context.resolver.resolve_expr(cm), optional_vars=ast.Name(id=name) if name is not None else None
            )
            for cm, name in zip(self.__cms, self.__names)
        ]

        stmt = (
            ast.AsyncWith(
                items=items,
                body=self._context.current_body,
                lineno=0,
            )
            if self.__is_async
            else ast.With(
                items=items,
                body=self._context.current_body,
                lineno=0,
            )
        )

        return [stmt]

    @override
    def _create_scope_builder(self) -> _BaseScopeBodyASTBuilder:
        return _BaseScopeBodyASTBuilder(self._context, self)


# noinspection PyTypeChecker
class IfStatementASTBuilder(_BaseHeaderASTBuilder["IfStatementASTBuilder"]):
    class _Body(_BaseChainPartASTBuilder):
        pass

    class _Else(_BaseChainPartASTBuilder):
        pass

    def __init__(self, context: BuildContext, test: Expr) -> None:
        super().__init__(context)
        self.__test = test
        self.__body = self._Body(context)
        self.__else: t.Optional[IfStatementASTBuilder._Else] = None

    def body(self) -> t.ContextManager[ScopeASTBuilder]:
        return self.__body

    def else_(self) -> t.ContextManager[ScopeASTBuilder]:
        part = self._Else(self._context)
        self.__else = part
        return part

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        return [
            ast.If(
                test=self._context.resolver.resolve_expr(self.__test),
                body=self._context.resolver.resolve_stmts(self.__body),
                orelse=self._context.resolver.resolve_stmts(self.__else),
                lineno=0,
            ),
        ]

    @override
    def _create_scope_builder(self) -> Self:
        return self


# noinspection PyTypeChecker
class TryStatementASTBuilder(_BaseHeaderASTBuilder["TryStatementASTBuilder"]):
    class _Body(_BaseChainPartASTBuilder):
        pass

    class _Except(_BaseChainPartASTBuilder):
        def __init__(
            self,
            context: BuildContext,
            types: t.Sequence[TypeRef],
            name: t.Optional[str],
        ) -> None:
            super().__init__(context)
            self.__types = types
            self.__name = name

        def handler(self) -> ast.ExceptHandler:
            base = _BaseASTBuilder(self._context)

            return ast.ExceptHandler(
                type=base.tuple_expr(*self.__types),
                name=self.__name,
                body=self._context.resolver.resolve_stmts(self, pass_if_empty=True),
            )

    class _Else(_BaseChainPartASTBuilder):
        pass

    class _Finally(_BaseChainPartASTBuilder):
        pass

    def __init__(self, context: BuildContext) -> None:
        super().__init__(context)
        self.__body = self._Body(context)
        self.__excepts = list[TryStatementASTBuilder._Except]()
        self.__else: t.Optional[TryStatementASTBuilder._Else] = None
        self.__finally: t.Optional[TryStatementASTBuilder._Finally] = None

    def body(self) -> t.ContextManager[ScopeASTBuilder]:
        return self.__body

    def except_(self, *types: TypeRef, name: t.Optional[str] = None) -> t.ContextManager[ScopeASTBuilder]:
        part = self._Except(self._context, types, name)
        self.__excepts.append(part)
        return part

    def else_(self) -> t.ContextManager[ScopeASTBuilder]:
        part = self._Else(self._context)
        self.__else = part
        return part

    def finally_(self) -> t.ContextManager[ScopeASTBuilder]:
        part = self._Finally(self._context)
        self.__finally = part
        return part

    @override
    def build_stmt(self) -> t.Sequence[ast.stmt]:
        if not self.__excepts and self.__finally is None:
            msg = "invalid try-except-finally block"
            raise RuntimeError(msg, self)

        return [
            ast.Try(
                body=self._context.resolver.resolve_stmts(self.__body),
                handlers=[part.handler() for part in self.__excepts],
                orelse=self._context.resolver.resolve_stmts(self.__else),
                finalbody=self._context.resolver.resolve_stmts(self.__finally),
            ),
        ]

    @override
    def _create_scope_builder(self) -> Self:
        return self


class ClassRefBuilder(ASTExpressionBuilder):
    def __init__(self, context: BuildContext, info: TypeInfo) -> None:
        self.__context = context
        self.__info = info
        self.__transforms: t.Callable[[TypeInfo], ast.expr] = self.__context.resolver.resolve_expr
        self.__base = _BaseASTBuilder(context)

    def optional(self) -> ClassRefBuilder:
        inner = self.__transforms

        def transform(info: TypeInfo) -> ast.expr:
            return self.__base.optional_type(inner(info))

        self.__transforms = transform

        return self

    def set(self) -> ClassRefBuilder:
        inner = self.__transforms

        def transform(info: TypeInfo) -> ast.expr:
            return self.__base.generic_type(set, inner(info))

        self.__transforms = transform

        return self

    def list(self) -> ClassRefBuilder:
        inner = self.__transforms

        def transform(info: TypeInfo) -> ast.expr:
            return self.__base.generic_type(list, inner(info))

        self.__transforms = transform

        return self

    def dict_value(self, key: TypeRef) -> ClassRefBuilder:
        inner = self.__transforms

        def transform(info: TypeInfo) -> ast.expr:
            return self.__base.generic_type(dict, key, inner(info))

        self.__transforms = transform

        return self

    def context_manager(self, *, is_async: bool = False) -> ClassRefBuilder:
        inner = self.__transforms

        def transform(info: TypeInfo) -> ast.expr:
            return self.__base.context_manager_type(inner(info), is_async=is_async)

        self.__transforms = transform

        return self

    def iterator(self, *, is_async: bool = False) -> ClassRefBuilder:
        inner = self.__transforms

        def transform(info: TypeInfo) -> ast.expr:
            return self.__base.iterator_type(inner(info), is_async=is_async)

        self.__transforms = transform

        return self

    def iterable(self, *, is_async: bool = False) -> ClassRefBuilder:
        inner = self.__transforms

        def transform(info: TypeInfo) -> ast.expr:
            return self.__base.iterable_type(inner(info), is_async=is_async)

        self.__transforms = transform

        return self

    def attr(self, *tail: str) -> AttrASTBuilder:
        return AttrASTBuilder(self.__context, self, *tail)

    def init(
        self,
        args: t.Optional[t.Sequence[Expr]] = None,
        kwargs: t.Optional[t.Mapping[str, Expr]] = None,
    ) -> CallASTBuilder:
        return CallASTBuilder(self.__context, self, args, kwargs)

    @override
    def build_expr(self) -> ast.expr:
        return self.__transforms(self.__info)


class ClassBodyASTBuilder(_BaseScopeBodyASTBuilder, TypeDefinitionBuilder):
    def __init__(self, context: BuildContext, header: ClassHeaderASTBuilder) -> None:
        super().__init__(context, header)
        self.__header = header

    @override
    @property
    def info(self) -> TypeInfo:
        return self.__header.info

    @override
    def ref(self) -> ClassRefBuilder:
        return self.__header.ref()

    def method_def(self, name: str) -> MethodHeaderASTBuilder:
        return MethodHeaderASTBuilder(self._context, name)

    def new_def(self) -> MethodHeaderASTBuilder:
        return self.method_def("__new__")

    def init_def(self) -> MethodHeaderASTBuilder:
        return self.method_def("__init__").returns(self.const(None))

    @contextmanager
    def init_self_attrs_def(self, attrs: t.Mapping[str, TypeRef]) -> t.Iterator[MethodBodyASTBuilder]:
        init_def = self.init_def()

        for name, annotation in attrs.items():
            init_def.arg(name=name, annotation=annotation)

        with init_def as init_body:
            for name in attrs:
                init_body.assign_stmt(init_body.self_attr(name), init_body.attr(name))

            yield init_body

    def call_def(self) -> MethodHeaderASTBuilder:
        return self.method_def("__call__")

    def str_def(self) -> MethodHeaderASTBuilder:
        return self.method_def("__str__").returns(str)

    def repr_def(self) -> MethodHeaderASTBuilder:
        return self.method_def("__repr__").returns(str)

    def hash_def(self) -> MethodHeaderASTBuilder:
        return self.method_def("__hash__").returns(bool)

    def eq_def(self) -> MethodHeaderASTBuilder:
        return self.method_def("__eq__").returns(bool)

    def ne_def(self) -> MethodHeaderASTBuilder:
        return self.method_def("__ne__").returns(bool)

    def ge_def(self) -> MethodHeaderASTBuilder:
        return self.method_def("__ge__").returns(bool)

    def gt_def(self) -> MethodHeaderASTBuilder:
        return self.method_def("__gt__").returns(bool)

    def le_def(self) -> MethodHeaderASTBuilder:
        return self.method_def("__le__").returns(bool)

    def lt_def(self) -> MethodHeaderASTBuilder:
        return self.method_def("__lt__").returns(bool)

    def property_getter_def(self, name: str) -> FuncHeaderASTBuilder:
        return self.func_def(name).arg("self").decorators(predef().property)

    def property_setter_def(self, name: str) -> FuncHeaderASTBuilder:
        return self.func_def(name).arg("self").decorators(self.attr(name, "setter"))


@dataclass(frozen=True)
class TypeVar:
    name: str
    bound: t.Optional[TypeRef]


# noinspection PyTypeChecker
class ClassHeaderASTBuilder(_BaseHeaderASTBuilder[ClassBodyASTBuilder], TypeDefinitionBuilder):
    def __init__(self, context: BuildContext, name: str) -> None:
        super().__init__(context, name)
        self.__info = NamedTypeInfo(name=name, module=self._context.module, namespace=self._context.namespace)
        self.__bases = list[TypeRef]()
        self.__decorators = list[TypeRef]()
        self.__keywords = dict[str, TypeRef]()
        self.__type_vars = list[TypeVar]()
        self.__docs = list[str]()

    @override
    @property
    def info(self) -> TypeInfo:
        return self.__info

    @override
    def ref(self) -> ClassRefBuilder:
        return ClassRefBuilder(self._context, self.__info)

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

        @override
        def build_stmt(self) -> t.Sequence[ast.stmt]:
            # noinspection PyUnresolvedReferences,PyArgumentList
            return (
                ast.ClassDef(
                    name=self._context.name,
                    bases=[self._context.resolver.resolve_expr(base) for base in self.__bases],
                    keywords=[
                        ast.keyword(
                            arg=key,
                            value=self._context.resolver.resolve_expr(value),
                        )
                        for key, value in self.__keywords.items()
                    ],
                    body=self._context.resolver.resolve_stmts(
                        *self._context.current_body, docs=self.__docs, pass_if_empty=True
                    ),
                    decorator_list=self.__build_decorators(),
                    type_params=[
                        ast.TypeVar(
                            name=type_var.name,
                            bound=self._context.resolver.resolve_expr(type_var.bound)
                            if type_var.bound is not None
                            else None,
                        )
                        for type_var in self.__type_vars
                    ],
                ),
            )

    else:

        @override
        def build(self) -> t.Sequence[ast.stmt]:
            return (
                ast.ClassDef(
                    name=self._context.name,
                    bases=[self._context.resolver.resolve_expr(base) for base in self.__bases],
                    keywords=[
                        ast.keyword(
                            arg=key,
                            value=self._context.resolver.resolve_expr(value),
                        )
                        for key, value in self.__keywords.items()
                    ],
                    body=self._context.resolver.resolve_stmts(
                        *self._context.current_body, docs=self.__docs, pass_if_empty=True
                    ),
                    decorator_list=self.__build_decorators(),
                ),
            )

    @override
    def _create_scope_builder(self) -> ClassBodyASTBuilder:
        return ClassBodyASTBuilder(self._context, self)

    def __build_decorators(self) -> list[ast.expr]:
        return [self._context.resolver.resolve_expr(dec) for dec in self.__decorators]


# noinspection PyTypeChecker,PyAbstractClass
class _BaseFuncSignatureASTBuilder(_BaseHeaderASTBuilder[T_co]):
    def __init__(self, context: BuildContext, name: str) -> None:
        super().__init__(context, name)
        self.__name = name
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

        if self.__is_async:
            node = ast.AsyncFunctionDef(  # type: ignore[call-overload,no-any-return,unused-ignore]
                # type_comment and type_params has default value each in 3.12 and not available in 3.9
                name=self.__name,
                args=self.__build_args(),
                decorator_list=self.__build_decorators(),
                returns=self.__build_returns(),
                body=self.__build_body(),
                lineno=0,
            )

        else:
            node = ast.FunctionDef(  # type: ignore[call-overload,no-any-return,unused-ignore]
                # type_comment and type_params has default value each in 3.12 and not available in 3.9
                name=self.__name,
                decorator_list=self.__build_decorators(),
                args=self.__build_args(),
                body=self.__build_body(),
                returns=self.__build_returns(),
                lineno=0,
            )

        return (node,)

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

        return [
            self._context.resolver.resolve_expr(dec)
            for dec in chain(head_decorators, self.__decorators, last_decorators)
        ]

    def __build_args(self) -> ast.arguments:
        return ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(
                    arg=arg,
                    annotation=self._context.resolver.resolve_expr(annotation) if annotation is not None else None,
                )
                for arg, annotation in self.__args
            ],
            defaults=[
                self._context.resolver.resolve_expr(self.__defaults[arg])
                for arg, _ in self.__args
                if arg in self.__defaults
            ],
            kwonlyargs=[
                ast.arg(
                    arg=arg,
                    annotation=self._context.resolver.resolve_expr(annotation) if annotation is not None else None,
                )
                for arg, annotation in self.__kwargs.items()
            ],
            kw_defaults=[
                self._context.resolver.resolve_expr(self.__defaults[key])
                for key in self.__kwargs
                if key in self.__defaults
            ],
        )

    def __build_returns(self) -> t.Optional[ast.expr]:
        if self.__returns is None:
            return None

        ret = self._context.resolver.resolve_expr(self.__returns)
        if self.__iterator_cm:
            ret = _BaseASTBuilder(self._context).iterator_type(ret, is_async=self.__is_async)

        return ret

    def __build_body(self) -> list[ast.stmt]:
        body: t.Sequence[Stmt]

        if self.__is_stub:
            body = [ast.Expr(value=ast.Constant(value=...))]

        elif self.__is_not_implemented:
            body = [ast.Raise(exc=ast.Name(id="NotImplementedError"))]

        else:
            body = self._context.current_body

        return self._context.resolver.resolve_stmts(*body, docs=self.__docs, pass_if_empty=True)


class FuncBodyASTBuilder(_BaseScopeBodyASTBuilder):
    pass


class FuncHeaderASTBuilder(_BaseFuncSignatureASTBuilder[FuncBodyASTBuilder]):
    @override
    def _create_scope_builder(self) -> FuncBodyASTBuilder:
        return FuncBodyASTBuilder(self._context, self)


class MethodBodyASTBuilder(FuncBodyASTBuilder):
    def self_attr(self, head: str, *tail: str) -> AttrASTBuilder:
        return self.attr("self", f"__{head}", *tail)


class MethodHeaderASTBuilder(_BaseFuncSignatureASTBuilder[MethodBodyASTBuilder]):
    def __init__(self, context: BuildContext, name: str) -> None:
        super().__init__(context, name)
        self.arg("self")

    @override
    def _create_scope_builder(self) -> MethodBodyASTBuilder:
        return MethodBodyASTBuilder(self._context, self)


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
    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        exc_traceback: object,
    ) -> None:
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

    def write(
        self,
        *,
        mode: t.Literal["w", "a"] = "w",
        mkdir: bool = False,
        exist_ok: bool = False,
    ) -> None:
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
    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        exc_traceback: object,
    ) -> None:
        if exc_type is not None:
            return

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

    def write(
        self,
        *,
        mode: t.Literal["w", "a"] = "w",
        mkdir: bool = False,
        exist_ok: bool = False,
    ) -> None:
        for builder in self.iter_modules():
            builder.write(mode=mode, mkdir=mkdir, exist_ok=exist_ok)
