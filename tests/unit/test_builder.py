import ast
import inspect
import typing as t

import pytest
from _pytest.mark import ParameterSet

from astlab.builder import Comprehension, ModuleASTBuilder, build_module, build_package
from astlab.reader import parse_module

_PARAMS: t.Final[list[ParameterSet]] = []


def _to_module_param(func: t.Callable[[], ModuleASTBuilder]) -> ParameterSet:
    expected_code = inspect.getdoc(func)
    assert expected_code is not None

    param = pytest.param(func(), parse_module(expected_code), id=func.__name__)
    _PARAMS.append(param)

    return param


@pytest.mark.parametrize(("builder", "expected"), _PARAMS)
def test_module_build(builder: ModuleASTBuilder, expected: ast.Module) -> None:
    assert builder.render() == ast.unparse(expected)


@_to_module_param
def build_empty_module() -> ModuleASTBuilder:
    """"""
    with build_module("simple") as mod:
        return mod


@_to_module_param
def build_simple_module() -> ModuleASTBuilder:
    # noinspection PySingleQuotedDocstring
    '''
    import abc
    import builtins
    import dataclasses
    import typing

    class Foo:
        """Docstring Foo."""

        @dataclasses.dataclass()
        class Bar:
            spam: builtins.int

        bars: typing.Optional[builtins.list[Bar]]

        def __init__(self, my_bar: Bar) -> None:
            self.__my_bar = my_bar

        def do_stuff(self, x: builtins.int) -> builtins.str:
            """Docstring do_stuff."""
            self.__some = Foo.Bar(x=x)
            x_str = builtins.str(x)
            return y.__str__()

        @abc.abstractmethod
        def do_buzz(self) -> builtins.object:
            raise NotImplementedError
    '''

    with (
        build_module("simple") as mod,
        mod.class_def("Foo").docstring("Docstring Foo.") as foo,
    ):
        with mod.class_def("Bar").dataclass() as bar:
            mod.field_def("spam", int)

        mod.field_def("bars", bar.ref().list().optional())

        with foo.init_self_attrs_def({"my_bar": bar}):
            pass

        with foo.method_def("do_stuff").arg("x", int).returns(str).docstring("Docstring do_stuff."):
            mod.assign_stmt(mod.attr("self", "__some"), bar.ref().init().kwarg("x", mod.attr("x")))
            mod.assign_stmt("x_str", mod.call(str, [mod.attr("x")]))
            mod.return_stmt(mod.attr("y").attr("__str__").call())

        with foo.method_def("do_buzz").abstract().returns(object).not_implemented():
            pass

        return mod


@_to_module_param
def build_bar_impl_module() -> ModuleASTBuilder:
    """
    import builtins
    import contextlib
    import simple.foo
    import typing

    class Bar(simple.foo.Foo):
        @typing.override
        @contextlib.contextmanager
        def do_stuff(self, spam: builtins.str) -> typing.Iterator[builtins.str]:
            ...
    """

    with (
        build_package("simple") as pkg,
        pkg.module("foo") as foo,
        foo.class_def("Foo") as foo_class,
        foo_class.method_def("do_stuff")
        .arg("spam", str)
        .returns(foo.type_ref(str).context_manager())
        .abstract()
        .not_implemented(),
    ):
        pass

    with (
        pkg.module("bar") as bar,
        bar.class_def("Bar").inherits(foo_class) as bar_cls,
        bar_cls.method_def("do_stuff").arg("spam", str).returns(str).context_manager().overrides().stub(),
    ):
        return bar


@_to_module_param
def build_optionals() -> ModuleASTBuilder:
    """
    import builtins
    import typing

    class MyOptions:
        my_generic_option_int: typing.Optional[builtins.int]
        my_optional_str: typing.Optional[builtins.str]
        my_optional_list_of_int: typing.Optional[builtins.list[builtins.int]]
        my_list_of_optional_int: builtins.list[typing.Optional[builtins.int]]
    """

    with build_module("opts") as mod, mod.class_def("MyOptions") as opt:
        opt.field_def("my_generic_option_int", opt.generic_type(t.Optional, int))
        opt.field_def("my_optional_str", opt.type_ref(str).optional())
        opt.field_def("my_optional_list_of_int", opt.type_ref(int).list().optional())
        opt.field_def("my_list_of_optional_int", opt.type_ref(int).optional().list())

        return mod


@_to_module_param
def build_runtime_types() -> ModuleASTBuilder:
    """
    import builtins
    import typing

    int_to_str = builtins.dict[builtins.int, builtins.str]()
    int_to_opt_str = builtins.dict[builtins.int, typing.Optional[builtins.str]]()
    """

    with build_module("types") as mod:
        mod.assign_stmt("int_to_str", mod.type_ref(dict[int, str]).init())
        mod.assign_stmt("int_to_opt_str", mod.type_ref(dict[int, t.Optional[str]]).init())

        return mod


@_to_module_param
def build_is_not_none_expr() -> ModuleASTBuilder:
    """
    maybe = body if test is not None else None
    """

    with build_module("types") as mod:
        mod.assign_stmt("maybe", mod.ternary_not_none_expr(mod.attr("body"), mod.attr("test")))

        return mod


@_to_module_param
def build_list_const() -> ModuleASTBuilder:
    """
    result = [1, 2, foo, bar]
    """

    with build_module("types") as mod:
        mod.assign_stmt("result", mod.list_expr([mod.const(1), mod.const(2), mod.attr("foo"), mod.attr("bar")]))

        return mod


@_to_module_param
def build_list_compr_expr() -> ModuleASTBuilder:
    """
    result = [item for target in iterable]
    """

    with build_module("types") as mod:
        mod.assign_stmt(
            "result",
            mod.list_expr(Comprehension(mod.attr("target"), mod.attr("iterable")), mod.attr("item")),
        )

        return mod


@_to_module_param
def build_try_except_else() -> ModuleASTBuilder:
    """
    import builtins

    try:
        1 / 0
    except builtins.ZeroDivisionError:
        print("zero division error")
    else:
        print("else")
    finally:
        print("finally")
    """

    with build_module("zero") as mod:
        with mod.try_stmt() as try_stmt:
            with try_stmt.body() as try_scope:
                (try_scope.const(1) / try_scope.const(0)).stmt()

            with try_stmt.except_(ZeroDivisionError) as except_scope:
                except_scope.attr("print").call().arg(except_scope.const("zero division error")).stmt()

            with try_stmt.else_() as else_scope:
                else_scope.stmt(else_scope.attr("print").call().arg(else_scope.const("else")))

            with try_stmt.finally_() as fin_scope:
                fin_scope.stmt(fin_scope.attr("print").call().arg(fin_scope.const("finally")))

        return mod
