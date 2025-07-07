import math
import typing as t
from pathlib import Path
from types import ModuleType

import pytest

from astlab import abc as astlab_abc
from astlab.types.annotator import TypeAnnotator
from astlab.types.inspector import TypeInspector
from astlab.types.loader import TypeLoader
from astlab.types.model import (
    LiteralTypeInfo,
    ModuleInfo,
    NamedTypeInfo,
    PackageInfo,
    RuntimeType,
    TypeInfo,
    builtins_module_info,
    none_type_info,
)
from tests.stub.types import StubBar, StubFoo, StubInt, StubX


class TestPackageInfo:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                [],
                None,
            ),
            pytest.param(
                ["pyprotostuben"],
                PackageInfo("pyprotostuben"),
            ),
            pytest.param(
                ["pyprotostuben", "python"],
                PackageInfo("python", PackageInfo("pyprotostuben")),
            ),
        ],
    )
    def test_build_or_none_ok(self, value: t.Sequence[str], expected: t.Optional[PackageInfo]) -> None:
        assert PackageInfo.build_or_none(*value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                PackageInfo("pyprotostuben"),
                None,
            ),
            pytest.param(
                PackageInfo("python", PackageInfo("pyprotostuben")),
                PackageInfo("pyprotostuben"),
            ),
        ],
    )
    def test_parent_ok(self, value: PackageInfo, expected: t.Optional[PackageInfo]) -> None:
        assert value.parent == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                PackageInfo("pyprotostuben"),
                Path("pyprotostuben"),
            ),
            pytest.param(
                PackageInfo("python", PackageInfo("pyprotostuben")),
                Path("pyprotostuben") / "python",
            ),
        ],
    )
    def test_directory_ok(self, value: PackageInfo, expected: Path) -> None:
        assert value.directory == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                PackageInfo("pyprotostuben"),
                ("pyprotostuben",),
            ),
            pytest.param(
                PackageInfo("python", PackageInfo("pyprotostuben")),
                ("pyprotostuben", "python"),
            ),
        ],
    )
    def test_parts_ok(self, value: PackageInfo, expected: t.Sequence[str]) -> None:
        assert value.parts == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                PackageInfo("pyprotostuben"),
                "pyprotostuben",
            ),
            pytest.param(
                PackageInfo("python", PackageInfo("pyprotostuben")),
                "pyprotostuben.python",
            ),
        ],
    )
    def test_qualname_ok(self, value: PackageInfo, expected: t.Sequence[str]) -> None:
        assert value.qualname == expected


class TestModuleInfo:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                "builtins",
                ModuleInfo("builtins"),
            ),
            pytest.param(
                "pyprotostuben.python.info",
                ModuleInfo("info", PackageInfo("python", PackageInfo("pyprotostuben"))),
            ),
        ],
    )
    def test_from_str_ok(self, value: str, expected: ModuleInfo) -> None:
        assert ModuleInfo.from_str(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                math,
                ModuleInfo("math"),
            ),
            pytest.param(
                astlab_abc,
                ModuleInfo("abc", PackageInfo("astlab")),
            ),
        ],
    )
    def test_from_module_ok(self, value: ModuleType, expected: ModuleInfo) -> None:
        assert ModuleInfo.from_module(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                ModuleInfo("math"),
                None,
            ),
            pytest.param(
                ModuleInfo("abc", PackageInfo("codegen", PackageInfo("pyprotostuben"))),
                PackageInfo("codegen", PackageInfo("pyprotostuben")),
            ),
        ],
    )
    def test_package_ok(self, value: ModuleInfo, expected: t.Optional[PackageInfo]) -> None:
        assert value.package == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                ModuleInfo("math"),
                Path("math.py"),
            ),
            pytest.param(
                ModuleInfo("abc", PackageInfo("codegen", PackageInfo("pyprotostuben"))),
                Path("pyprotostuben") / "codegen" / "abc.py",
            ),
        ],
    )
    def test_file_ok(self, value: ModuleInfo, expected: Path) -> None:
        assert value.file == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                ModuleInfo("math"),
                Path("math.pyi"),
            ),
            pytest.param(
                ModuleInfo("abc", PackageInfo("codegen", PackageInfo("pyprotostuben"))),
                Path("pyprotostuben") / "codegen" / "abc.pyi",
            ),
        ],
    )
    def test_stub_file_ok(self, value: ModuleInfo, expected: Path) -> None:
        assert value.stub_file == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                ModuleInfo("math"),
                ("math",),
            ),
            pytest.param(
                ModuleInfo("abc", PackageInfo("codegen", PackageInfo("pyprotostuben"))),
                ("pyprotostuben", "codegen", "abc"),
            ),
        ],
    )
    def test_parts_ok(self, value: ModuleInfo, expected: t.Sequence[str]) -> None:
        assert value.parts == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                ModuleInfo("math"),
                "math",
            ),
            pytest.param(
                ModuleInfo("abc", PackageInfo("codegen", PackageInfo("pyprotostuben"))),
                "pyprotostuben.codegen.abc",
            ),
        ],
    )
    def test_qualname_ok(self, value: ModuleInfo, expected: t.Sequence[str]) -> None:
        assert value.qualname == expected


TYPES_CASES = pytest.mark.parametrize(
    ("type_", "annotation", "info"),
    [
        pytest.param(
            int,
            "builtins.int",
            NamedTypeInfo("int", ModuleInfo("builtins")),
        ),
        pytest.param(
            t.Literal["foo", "bar", "baz"],
            "typing.Literal['foo', 'bar', 'baz']",
            LiteralTypeInfo(values=("foo", "bar", "baz")),
        ),
        pytest.param(
            t.Optional,
            "typing.Optional",
            NamedTypeInfo(
                name="Optional",
                module=ModuleInfo("typing"),
            ),
        ),
        pytest.param(
            t.Optional[int],
            "typing.Optional[builtins.int]",
            NamedTypeInfo(
                name="Optional",
                module=ModuleInfo("typing"),
                type_params=(NamedTypeInfo("int", ModuleInfo("builtins")),),
            ),
        ),
        pytest.param(
            t.Union[int, str],
            "typing.Union[builtins.int, builtins.str]",
            NamedTypeInfo(
                name="Union",
                module=ModuleInfo("typing"),
                type_params=(
                    NamedTypeInfo("int", ModuleInfo("builtins")),
                    NamedTypeInfo("str", ModuleInfo("builtins")),
                ),
            ),
        ),
        pytest.param(
            t.Union[int, str, None],
            "typing.Union[builtins.int, builtins.str, None]",
            NamedTypeInfo(
                name="Union",
                module=ModuleInfo("typing"),
                type_params=(
                    NamedTypeInfo("int", ModuleInfo("builtins")),
                    NamedTypeInfo("str", builtins_module_info()),
                    none_type_info(),
                ),
            ),
        ),
        pytest.param(
            t.Mapping[int, str],
            "typing.Mapping[builtins.int, builtins.str]",
            NamedTypeInfo(
                name="Mapping",
                module=ModuleInfo("typing"),
                type_params=(
                    NamedTypeInfo("int", builtins_module_info()),
                    NamedTypeInfo("str", builtins_module_info()),
                ),
            ),
        ),
        pytest.param(
            t.Mapping[int, t.Optional[str]],
            "typing.Mapping[builtins.int, typing.Optional[builtins.str]]",
            NamedTypeInfo(
                name="Mapping",
                module=ModuleInfo("typing"),
                type_params=(
                    NamedTypeInfo("int", builtins_module_info()),
                    NamedTypeInfo(
                        name="Optional",
                        module=ModuleInfo("typing"),
                        type_params=(NamedTypeInfo("str", builtins_module_info()),),
                    ),
                ),
            ),
        ),
        pytest.param(
            StubFoo,
            "tests.stub.types.StubFoo",
            NamedTypeInfo("StubFoo", ModuleInfo("types", PackageInfo("stub", PackageInfo("tests")))),
        ),
        pytest.param(
            StubBar,
            "tests.stub.types.StubBar",
            NamedTypeInfo("StubBar", ModuleInfo("types", PackageInfo("stub", PackageInfo("tests")))),
        ),
        pytest.param(
            StubBar[StubFoo],
            "tests.stub.types.StubBar[tests.stub.types.StubFoo]",
            NamedTypeInfo(
                name="StubBar",
                module=ModuleInfo("types", PackageInfo("stub", PackageInfo("tests"))),
                type_params=(NamedTypeInfo("StubFoo", ModuleInfo("types", PackageInfo("stub", PackageInfo("tests")))),),
            ),
        ),
        pytest.param(
            StubX.Y.Z,
            "tests.stub.types.StubX.Y.Z",
            NamedTypeInfo(
                name="Z",
                module=ModuleInfo("types", PackageInfo("stub", PackageInfo("tests"))),
                namespace=("StubX", "Y"),
            ),
        ),
        pytest.param(
            StubInt,
            "tests.stub.types.StubInt",
            NamedTypeInfo(
                name="StubInt",
                module=ModuleInfo("types", PackageInfo("stub", PackageInfo("tests"))),
            ),
        ),
    ],
)


class TestTypeAnnotator:
    @TYPES_CASES
    def test_parse_ok(
        self,
        type_annotator: TypeAnnotator,
        type_: RuntimeType,
        annotation: str,
        info: TypeInfo,
    ) -> None:
        assert type_annotator.parse(annotation) == info

    @TYPES_CASES
    def test_annotate_ok(
        self,
        type_annotator: TypeAnnotator,
        type_: RuntimeType,
        annotation: str,
        info: TypeInfo,
    ) -> None:
        assert type_annotator.annotate(type_annotator.parse(annotation)) == annotation


class TestTypeInspector:
    @TYPES_CASES
    def test_inspect_ok(
        self,
        type_inspector: TypeInspector,
        type_: RuntimeType,
        annotation: str,
        info: TypeInfo,
    ) -> None:
        assert type_inspector.inspect(type_) == info


class TestTypeLoader:
    @TYPES_CASES
    def test_load_ok(
        self,
        type_loader: TypeLoader,
        type_inspector: TypeInspector,
        type_: RuntimeType,
        annotation: str,
        info: TypeInfo,
    ) -> None:
        loaded = type_loader.load(info)

        assert loaded == type_
        assert t.get_origin(loaded) is t.get_origin(type_)
        assert t.get_args(loaded) == t.get_args(type_)

    @pytest.mark.parametrize(
        ("info", "error"),
        [
            pytest.param(
                NamedTypeInfo("NonExistingType", builtins_module_info()),
                AttributeError,
            ),
            pytest.param(
                NamedTypeInfo("SomeType", ModuleInfo("non_existing_module")),
                ModuleNotFoundError,
            ),
        ],
    )
    def test_load_error(self, type_loader: TypeLoader, info: TypeInfo, error: type[Exception]) -> None:
        with pytest.raises(error):
            type_loader.load(info)
