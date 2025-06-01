import math
import typing as t
from pathlib import Path
from types import ModuleType

import pytest

from astlab import abc as astlab_abc
from astlab.info import ModuleInfo, PackageInfo, RuntimeType, TypeInfo, builtins_module, none_type_info


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


class TestTypeInfo:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                "builtins.int",
                TypeInfo("int", ModuleInfo("builtins")),
            ),
            pytest.param(
                "tests.unit.test_info.TestTypeInfo",
                TypeInfo("TestTypeInfo", ModuleInfo("test_info", PackageInfo("unit", PackageInfo("tests")))),
            ),
            pytest.param(
                "typing.Optional",
                TypeInfo(
                    name="Optional",
                    module=ModuleInfo("typing"),
                ),
            ),
            pytest.param(
                "typing.Optional[builtins.int]",
                TypeInfo(
                    name="Optional",
                    module=ModuleInfo("typing"),
                    type_params=(TypeInfo("int", ModuleInfo("builtins")),),
                ),
            ),
            pytest.param(
                "typing.Union[builtins.int, builtins.str]",
                TypeInfo(
                    name="Union",
                    module=ModuleInfo("typing"),
                    type_params=(
                        TypeInfo("int", ModuleInfo("builtins")),
                        TypeInfo("str", ModuleInfo("builtins")),
                    ),
                ),
            ),
            pytest.param(
                "typing.Union[builtins.int, builtins.str, None]",
                TypeInfo(
                    name="Union",
                    module=ModuleInfo("typing"),
                    type_params=(
                        TypeInfo("int", ModuleInfo("builtins")),
                        TypeInfo("str", builtins_module()),
                        none_type_info(),
                    ),
                ),
            ),
            pytest.param(
                "typing.Mapping[builtins.int, builtins.str]",
                TypeInfo(
                    name="Mapping",
                    module=ModuleInfo("typing"),
                    type_params=(
                        TypeInfo("int", builtins_module()),
                        TypeInfo("str", builtins_module()),
                    ),
                ),
            ),
            pytest.param(
                "typing.Mapping[builtins.int, typing.Optional[builtins.str]]",
                TypeInfo(
                    name="Mapping",
                    module=ModuleInfo("typing"),
                    type_params=(
                        TypeInfo("int", builtins_module()),
                        TypeInfo(
                            name="Optional",
                            module=ModuleInfo("typing"),
                            type_params=(TypeInfo("str", builtins_module()),),
                        ),
                    ),
                ),
            ),
        ],
    )
    def test_from_str_ok(self, value: str, expected: TypeInfo) -> None:
        assert TypeInfo.from_str(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                int,
                TypeInfo("int", builtins_module()),
            ),
            pytest.param(
                TypeInfo,
                TypeInfo("TypeInfo", ModuleInfo("info", PackageInfo("astlab"))),
            ),
            pytest.param(
                t.Optional[int],
                TypeInfo(
                    name="Optional",
                    module=ModuleInfo("typing"),
                    type_params=(TypeInfo("int", builtins_module()),),
                ),
            ),
            pytest.param(
                t.Union[int, str],
                TypeInfo(
                    name="Union",
                    module=ModuleInfo("typing"),
                    type_params=(
                        TypeInfo("int", builtins_module()),
                        TypeInfo("str", builtins_module()),
                    ),
                ),
            ),
            pytest.param(
                t.Union[int, str, None],
                TypeInfo(
                    name="Union",
                    module=ModuleInfo("typing"),
                    type_params=(
                        TypeInfo("int", builtins_module()),
                        TypeInfo("str", builtins_module()),
                        none_type_info(),
                    ),
                ),
            ),
            pytest.param(
                t.Mapping[int, str],
                TypeInfo(
                    name="Mapping",
                    module=ModuleInfo("typing"),
                    type_params=(
                        TypeInfo("int", ModuleInfo("builtins")),
                        TypeInfo("str", ModuleInfo("builtins")),
                    ),
                ),
            ),
            pytest.param(
                t.Mapping[int, t.Optional[str]],
                TypeInfo(
                    name="Mapping",
                    module=ModuleInfo("typing"),
                    type_params=(
                        TypeInfo("int", ModuleInfo("builtins")),
                        TypeInfo(
                            name="Optional",
                            module=ModuleInfo("typing"),
                            type_params=(TypeInfo("str", ModuleInfo("builtins")),),
                        ),
                    ),
                ),
            ),
        ],
    )
    def test_from_type_ok(self, value: RuntimeType, expected: TypeInfo) -> None:
        assert TypeInfo.from_type(value) == expected

    @pytest.mark.parametrize(
        "annotation",
        [
            pytest.param("builtins.int"),
            pytest.param("tests.unit.test_info.TestTypeInfo"),
            pytest.param("typing.Optional"),
            pytest.param("typing.Optional[builtins.int]"),
            pytest.param("typing.Union[builtins.int, builtins.str]"),
            pytest.param("typing.Union[builtins.int, builtins.str, None]"),
            pytest.param("typing.Mapping[builtins.int, builtins.str]"),
            pytest.param("typing.Mapping[builtins.int, typing.Optional[builtins.str]]"),
        ],
    )
    def test_from_str_to_annotation_are_same(self, annotation: str) -> None:
        assert TypeInfo.from_str(annotation).annotation() == annotation
