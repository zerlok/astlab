import typing as t
from dataclasses import dataclass
from pathlib import Path

from astlab.types import (
    EnumTypeInfo,
    EnumTypeValue,
    LiteralTypeInfo,
    ModuleInfo,
    NamedTypeInfo,
    PackageInfo,
    TypeInfo,
    UnionTypeInfo,
)
from tests.marks import FEATURE_TYPE_ALIAS_QUALNAME, FEATURE_TYPING_UNION_IS_UNION_TYPE, FEATURE_UNION_TYPE_SYNTAX
from tests.stub.types import StubBar, StubCM, StubEnum, StubFoo, StubInt, StubNode, StubUnionAlias, StubUnionType, StubX


@dataclass(frozen=True)
class PackageCase:
    info: PackageInfo
    valid_directory: Path
    valid_parent: t.Optional[PackageInfo]
    valid_parts: t.Sequence[str]
    valid_qualname: str


@dataclass(frozen=True)
class ModuleCase:
    info: ModuleInfo
    valid_file: Path
    valid_package: t.Optional[PackageInfo]
    valid_parts: t.Sequence[str]
    valid_qualname: str


@dataclass(frozen=True)
class TypeCase:
    info: TypeInfo
    valid_annotation: str
    python_type: object


def simple_package() -> PackageCase:
    return PackageCase(
        info=PackageInfo("simple"),
        valid_directory=Path("simple"),
        valid_parent=None,
        valid_parts=("simple",),
        valid_qualname="simple",
    )


def some_sub_package() -> PackageCase:
    return PackageCase(
        info=PackageInfo("sub", PackageInfo("some")),
        valid_directory=Path("some") / "sub",
        valid_parent=PackageInfo("some"),
        valid_parts=("some", "sub"),
        valid_qualname="some.sub",
    )


def foo_bar_baz_package() -> PackageCase:
    return PackageCase(
        info=PackageInfo("baz", PackageInfo("bar", PackageInfo("foo"))),
        valid_directory=Path("foo") / "bar" / "baz",
        valid_parent=PackageInfo("bar", PackageInfo("foo")),
        valid_parts=("foo", "bar", "baz"),
        valid_qualname="foo.bar.baz",
    )


def abc_module() -> ModuleCase:
    return ModuleCase(
        info=ModuleInfo("abc"),
        valid_file=Path("abc.py"),
        valid_package=None,
        valid_parts=("abc",),
        valid_qualname="abc",
    )


def logging_config_module() -> ModuleCase:
    return ModuleCase(
        info=ModuleInfo("config", PackageInfo("logging")),
        valid_file=Path("logging") / "config.py",
        valid_package=PackageInfo("logging"),
        valid_parts=("logging", "config"),
        valid_qualname="logging.config",
    )


def none_type() -> TypeCase:
    return TypeCase(
        python_type=None,
        valid_annotation="None",
        info=NamedTypeInfo("NoneType", ModuleInfo("builtins")),
    )


def ellipsis_type() -> TypeCase:
    return TypeCase(
        python_type=...,
        valid_annotation="...",
        info=NamedTypeInfo("ellipsis", ModuleInfo("builtins")),
    )


def int_type() -> TypeCase:
    return TypeCase(
        python_type=int,
        valid_annotation="builtins.int",
        info=NamedTypeInfo("int", ModuleInfo("builtins")),
    )


def float_type() -> TypeCase:
    return TypeCase(
        python_type=float,
        valid_annotation="builtins.float",
        info=NamedTypeInfo("float", ModuleInfo("builtins")),
    )


def str_type() -> TypeCase:
    return TypeCase(
        python_type=str,
        valid_annotation="builtins.str",
        info=NamedTypeInfo("str", ModuleInfo("builtins")),
    )


def list_type() -> TypeCase:
    return TypeCase(
        python_type=list,
        valid_annotation="builtins.list",
        info=NamedTypeInfo("list", ModuleInfo("builtins")),
    )


def literal_foo_bar_baz_type() -> TypeCase:
    return TypeCase(
        python_type=t.Literal["foo", "bar", "baz"],
        valid_annotation="typing.Literal['foo', 'bar', 'baz']",
        info=LiteralTypeInfo(values=("foo", "bar", "baz")),
    )


def optional_type() -> TypeCase:
    return TypeCase(
        python_type=t.Optional,
        valid_annotation="typing.Optional",
        info=NamedTypeInfo("Optional", ModuleInfo("typing")),
    )


def optional_int_type() -> TypeCase:
    return TypeCase(
        python_type=t.Optional[int],
        valid_annotation="typing.Optional[builtins.int]",
        info=NamedTypeInfo(
            name="Optional",
            module=ModuleInfo("typing"),
            type_params=(NamedTypeInfo("int", ModuleInfo("builtins")),),
        ),
    )


def union_type() -> TypeCase:
    return TypeCase(
        python_type=t.Union,
        valid_annotation="typing.Union",
        info=NamedTypeInfo("Union", ModuleInfo("typing")),
    )


@FEATURE_TYPING_UNION_IS_UNION_TYPE.mark_obsolete()
def union_int_str_before_union_type_support() -> TypeCase:
    return TypeCase(
        python_type=t.Union[int, str],
        valid_annotation="typing.Union[builtins.int, builtins.str]",
        info=NamedTypeInfo(
            name="Union",
            module=ModuleInfo("typing"),
            type_params=(
                NamedTypeInfo("int", ModuleInfo("builtins")),
                NamedTypeInfo("str", ModuleInfo("builtins")),
            ),
        ),
    )


@FEATURE_TYPING_UNION_IS_UNION_TYPE.mark_required()
def union_int_str_with_union_type_support() -> TypeCase:
    return TypeCase(
        python_type=t.Union[int, str],
        valid_annotation="builtins.int | builtins.str",
        info=UnionTypeInfo(
            values=(
                NamedTypeInfo("int", ModuleInfo("builtins")),
                NamedTypeInfo("str", ModuleInfo("builtins")),
            ),
        ),
    )


@FEATURE_TYPING_UNION_IS_UNION_TYPE.mark_obsolete()
def union_int_str_none_before_union_type_support() -> TypeCase:
    return TypeCase(
        python_type=t.Union[int, str, None],
        valid_annotation="typing.Union[builtins.int, builtins.str, None]",
        info=NamedTypeInfo(
            name="Union",
            module=ModuleInfo("typing"),
            type_params=(
                NamedTypeInfo("int", ModuleInfo("builtins")),
                NamedTypeInfo("str", ModuleInfo("builtins")),
                NamedTypeInfo("NoneType", ModuleInfo("builtins")),
            ),
        ),
    )


@FEATURE_TYPING_UNION_IS_UNION_TYPE.mark_required()
def union_int_str_none_with_union_type_support() -> TypeCase:
    return TypeCase(
        python_type=t.Union[int, str, None],
        valid_annotation="builtins.int | builtins.str | None",
        info=UnionTypeInfo(
            values=(
                NamedTypeInfo("int", ModuleInfo("builtins")),
                NamedTypeInfo("str", ModuleInfo("builtins")),
                NamedTypeInfo("NoneType", ModuleInfo("builtins")),
            ),
        ),
    )


def mapping_int_str_type() -> TypeCase:
    return TypeCase(
        python_type=t.Mapping[int, str],
        valid_annotation="typing.Mapping[builtins.int, builtins.str]",
        info=NamedTypeInfo(
            name="Mapping",
            module=ModuleInfo("typing"),
            type_params=(
                NamedTypeInfo("int", ModuleInfo("builtins")),
                NamedTypeInfo("str", ModuleInfo("builtins")),
            ),
        ),
    )


def mapping_int_opt_str_type() -> TypeCase:
    return TypeCase(
        python_type=t.Mapping[int, t.Optional[str]],
        valid_annotation="typing.Mapping[builtins.int, typing.Optional[builtins.str]]",
        info=NamedTypeInfo(
            name="Mapping",
            module=ModuleInfo("typing"),
            type_params=(
                NamedTypeInfo("int", ModuleInfo("builtins")),
                NamedTypeInfo(
                    name="Optional",
                    module=ModuleInfo("typing"),
                    type_params=(NamedTypeInfo("str", ModuleInfo("builtins")),),
                ),
            ),
        ),
    )


def stub_foo_type() -> TypeCase:
    return TypeCase(
        python_type=StubFoo,
        valid_annotation="tests.stub.types.StubFoo",
        info=NamedTypeInfo("StubFoo", ModuleInfo("types", PackageInfo("stub", PackageInfo("tests")))),
    )


def stub_bar_type() -> TypeCase:
    return TypeCase(
        python_type=StubBar,
        valid_annotation="tests.stub.types.StubBar",
        info=NamedTypeInfo("StubBar", ModuleInfo("types", PackageInfo("stub", PackageInfo("tests")))),
    )


def stub_bar_foo_type() -> TypeCase:
    return TypeCase(
        python_type=StubBar[StubFoo],
        valid_annotation="tests.stub.types.StubBar[tests.stub.types.StubFoo]",
        info=NamedTypeInfo(
            name="StubBar",
            module=ModuleInfo("types", PackageInfo("stub", PackageInfo("tests"))),
            type_params=(NamedTypeInfo("StubFoo", ModuleInfo("types", PackageInfo("stub", PackageInfo("tests")))),),
        ),
    )


def stub_x_y_z_type() -> TypeCase:
    return TypeCase(
        python_type=StubX.Y.Z,
        valid_annotation="tests.stub.types.StubX.Y.Z",
        info=NamedTypeInfo(
            name="Z",
            module=ModuleInfo("types", PackageInfo("stub", PackageInfo("tests"))),
            namespace=("StubX", "Y"),
        ),
    )


def stub_enum_type() -> TypeCase:
    return TypeCase(
        python_type=StubEnum,
        valid_annotation="tests.stub.types.StubEnum",
        info=EnumTypeInfo(
            name="StubEnum",
            module=ModuleInfo("types", PackageInfo("stub", PackageInfo("tests"))),
            values=(
                EnumTypeValue(name="FOO", value=1),
                EnumTypeValue(name="BAR", value=2),
            ),
        ),
    )


def stub_cm_type() -> TypeCase:
    return TypeCase(
        python_type=StubCM,
        valid_annotation="tests.stub.types.StubCM",
        info=NamedTypeInfo(
            name="StubCM",
            module=ModuleInfo("types", PackageInfo("stub", PackageInfo("tests"))),
        ),
    )


def stub_node_type() -> TypeCase:
    return TypeCase(
        python_type=StubNode[int],
        valid_annotation="tests.stub.types.StubNode[builtins.int]",
        info=NamedTypeInfo(
            name="StubNode",
            module=ModuleInfo("types", PackageInfo("stub", PackageInfo("tests"))),
            type_params=(NamedTypeInfo("int", ModuleInfo("builtins")),),
        ),
    )


@FEATURE_TYPE_ALIAS_QUALNAME.mark_required()
def stub_alias_new_type() -> TypeCase:
    return TypeCase(
        python_type=StubInt,
        valid_annotation="tests.stub.types.StubInt",
        info=NamedTypeInfo(
            name="StubInt",
            module=ModuleInfo("types", PackageInfo("stub", PackageInfo("tests"))),
        ),
    )


@FEATURE_TYPE_ALIAS_QUALNAME.mark_required()
@FEATURE_TYPING_UNION_IS_UNION_TYPE.mark_obsolete()
def stub_union_alias_before_union_type_support() -> TypeCase:
    return TypeCase(
        python_type=StubUnionAlias,
        valid_annotation="typing.Union["
        "tests.stub.types.StubFoo, "
        "tests.stub.types.StubBar[tests.stub.types.StubInt], "
        "tests.stub.types.StubX"
        "]",
        info=NamedTypeInfo(
            name="Union",
            module=ModuleInfo(name="typing"),
            type_params=(
                NamedTypeInfo(
                    name="StubFoo",
                    module=ModuleInfo(
                        name="types",
                        package=PackageInfo(name="stub", parent=PackageInfo(name="tests")),
                    ),
                ),
                NamedTypeInfo(
                    name="StubBar",
                    module=ModuleInfo(
                        name="types",
                        package=PackageInfo(name="stub", parent=PackageInfo(name="tests")),
                    ),
                    type_params=(
                        NamedTypeInfo(
                            name="StubInt",
                            module=ModuleInfo(
                                name="types",
                                package=PackageInfo(name="stub", parent=PackageInfo(name="tests")),
                            ),
                        ),
                    ),
                ),
                NamedTypeInfo(
                    name="StubX",
                    module=ModuleInfo(
                        name="types",
                        package=PackageInfo(name="stub", parent=PackageInfo(name="tests")),
                    ),
                ),
            ),
        ),
    )


@FEATURE_TYPE_ALIAS_QUALNAME.mark_required()
@FEATURE_TYPING_UNION_IS_UNION_TYPE.mark_required()
def stub_union_alias_with_union_type_support() -> TypeCase:
    return TypeCase(
        python_type=StubUnionAlias,
        valid_annotation="tests.stub.types.StubFoo"
        " | tests.stub.types.StubBar[tests.stub.types.StubInt]"
        " | tests.stub.types.StubX",
        info=UnionTypeInfo(
            values=(
                NamedTypeInfo(
                    name="StubFoo",
                    module=ModuleInfo(
                        name="types",
                        package=PackageInfo(name="stub", parent=PackageInfo(name="tests")),
                    ),
                ),
                NamedTypeInfo(
                    name="StubBar",
                    module=ModuleInfo(
                        name="types",
                        package=PackageInfo(name="stub", parent=PackageInfo(name="tests")),
                    ),
                    type_params=(
                        NamedTypeInfo(
                            name="StubInt",
                            module=ModuleInfo(
                                name="types",
                                package=PackageInfo(name="stub", parent=PackageInfo(name="tests")),
                            ),
                        ),
                    ),
                ),
                NamedTypeInfo(
                    name="StubX",
                    module=ModuleInfo(
                        name="types",
                        package=PackageInfo(name="stub", parent=PackageInfo(name="tests")),
                    ),
                ),
            ),
        ),
    )


@FEATURE_UNION_TYPE_SYNTAX.mark_required()
def stub_union_type_with_union_syntax_support() -> TypeCase:
    return TypeCase(
        python_type=StubUnionType,
        valid_annotation="builtins.int | builtins.str | builtins.float | None",
        info=UnionTypeInfo(
            values=(
                NamedTypeInfo("int", ModuleInfo("builtins")),
                NamedTypeInfo("str", ModuleInfo("builtins")),
                NamedTypeInfo("float", ModuleInfo("builtins")),
                NamedTypeInfo("NoneType", ModuleInfo("builtins")),
            ),
        ),
    )
