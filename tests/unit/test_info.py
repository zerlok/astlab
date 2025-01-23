import math
import typing as t
from pathlib import Path
from types import ModuleType

import pytest

from astlab import abc as astlab_abc
from astlab.info import ModuleInfo, PackageInfo, TypeInfo


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
                PackageInfo(None, "pyprotostuben"),
            ),
            pytest.param(
                ["pyprotostuben", "python"],
                PackageInfo(PackageInfo(None, "pyprotostuben"), "python"),
            ),
        ],
    )
    def test_build_or_none_ok(self, value: t.Sequence[str], expected: t.Optional[PackageInfo]) -> None:
        assert PackageInfo.build_or_none(*value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                PackageInfo(None, "pyprotostuben"),
                None,
            ),
            pytest.param(
                PackageInfo(PackageInfo(None, "pyprotostuben"), "python"),
                PackageInfo(None, "pyprotostuben"),
            ),
        ],
    )
    def test_parent_ok(self, value: PackageInfo, expected: t.Optional[PackageInfo]) -> None:
        assert value.parent == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                PackageInfo(None, "pyprotostuben"),
                Path("pyprotostuben"),
            ),
            pytest.param(
                PackageInfo(PackageInfo(None, "pyprotostuben"), "python"),
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
                PackageInfo(None, "pyprotostuben"),
                ("pyprotostuben",),
            ),
            pytest.param(
                PackageInfo(PackageInfo(None, "pyprotostuben"), "python"),
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
                PackageInfo(None, "pyprotostuben"),
                "pyprotostuben",
            ),
            pytest.param(
                PackageInfo(PackageInfo(None, "pyprotostuben"), "python"),
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
                ModuleInfo(None, "builtins"),
            ),
            pytest.param(
                "pyprotostuben.python.info",
                ModuleInfo(PackageInfo(PackageInfo(None, "pyprotostuben"), "python"), "info"),
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
                ModuleInfo(None, "math"),
            ),
            pytest.param(
                astlab_abc,
                ModuleInfo(PackageInfo(None, "astlab"), "abc"),
            ),
        ],
    )
    def test_from_module_ok(self, value: ModuleType, expected: ModuleInfo) -> None:
        assert ModuleInfo.from_module(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                ModuleInfo(None, "math"),
                None,
            ),
            pytest.param(
                ModuleInfo(PackageInfo(PackageInfo(None, "pyprotostuben"), "codegen"), "abc"),
                PackageInfo(PackageInfo(None, "pyprotostuben"), "codegen"),
            ),
        ],
    )
    def test_package_ok(self, value: ModuleInfo, expected: t.Optional[PackageInfo]) -> None:
        assert value.package == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                ModuleInfo(None, "math"),
                Path("math.py"),
            ),
            pytest.param(
                ModuleInfo(PackageInfo(PackageInfo(None, "pyprotostuben"), "codegen"), "abc"),
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
                ModuleInfo(None, "math"),
                Path("math.pyi"),
            ),
            pytest.param(
                ModuleInfo(PackageInfo(PackageInfo(None, "pyprotostuben"), "codegen"), "abc"),
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
                ModuleInfo(None, "math"),
                ("math",),
            ),
            pytest.param(
                ModuleInfo(PackageInfo(PackageInfo(None, "pyprotostuben"), "codegen"), "abc"),
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
                ModuleInfo(None, "math"),
                "math",
            ),
            pytest.param(
                ModuleInfo(PackageInfo(PackageInfo(None, "pyprotostuben"), "codegen"), "abc"),
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
                "builtins:int",
                TypeInfo(ModuleInfo(None, "builtins"), ("int",)),
            ),
            pytest.param(
                "pyprotostuben.python.info:TypeInfo",
                TypeInfo(ModuleInfo(PackageInfo(PackageInfo(None, "pyprotostuben"), "python"), "info"), ("TypeInfo",)),
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
                TypeInfo(ModuleInfo(None, "builtins"), ("int",)),
            ),
            pytest.param(
                TypeInfo,
                TypeInfo(ModuleInfo(PackageInfo(None, "astlab"), "info"), ("TypeInfo",)),
            ),
        ],
    )
    def test_from_type_ok(self, value: type[object], expected: TypeInfo) -> None:
        assert TypeInfo.from_type(value) == expected
