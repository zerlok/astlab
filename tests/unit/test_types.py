import importlib
import typing as t

import pytest
from pytest_case_provider import inject_method

from astlab.types.annotator import TypeAnnotator
from astlab.types.inspector import TypeInspector
from astlab.types.loader import TypeLoader, TypeLoaderError
from astlab.types.model import (
    ModuleInfo,
    NamedTypeInfo,
    PackageInfo,
    TypeInfo,
    builtins_module_info,
)
from tests.unit.case_types import ModuleCase, PackageCase, TypeCase


class TestPackageInfo:
    @inject_method()
    def test_parent_is_valid(self, case: PackageCase) -> None:
        assert case.info.parent == case.valid_parent

    @inject_method()
    def test_directory_is_valid(self, case: PackageCase) -> None:
        assert case.info.directory == case.valid_directory

    @inject_method()
    def test_parts_are_valid(self, case: PackageCase) -> None:
        assert case.info.parts == case.valid_parts

    @inject_method()
    def test_qualname_is_valid(self, case: PackageCase) -> None:
        assert case.info.qualname == case.valid_qualname

    @inject_method()
    def test_build_or_none_from_parts_is_same_info(self, case: PackageCase) -> None:
        assert PackageInfo.build_or_none(*case.info.parts) == case.info

    @inject_method()
    def test_build_from_parts_is_same_info(self, case: PackageCase) -> None:
        assert PackageInfo.build(*case.info.parts) == case.info

    @inject_method()
    def test_from_qualname_str_is_same_info(self, case: PackageCase) -> None:
        assert PackageInfo.from_str(case.info.qualname) == case.info


class TestModuleInfo:
    @inject_method()
    def test_parent_is_valid(self, case: ModuleCase) -> None:
        assert case.info.package == case.valid_package

    @inject_method()
    def test_file_is_valid(self, case: ModuleCase) -> None:
        assert case.info.file == case.valid_file

    @inject_method()
    def test_stub_file_is_valid(self, case: ModuleCase) -> None:
        assert case.info.stub_file == case.valid_file.with_suffix(".pyi")

    @inject_method()
    def test_parts_are_valid(self, case: ModuleCase) -> None:
        assert case.info.parts == case.valid_parts

    @inject_method()
    def test_qualname_is_valid(self, case: ModuleCase) -> None:
        assert case.info.qualname == case.valid_qualname

    @inject_method()
    def test_build_or_none_from_parts_is_same_info(self, case: ModuleCase) -> None:
        assert ModuleInfo.build_or_none(*case.info.parts) == case.info

    @inject_method()
    def test_build_from_parts_is_same_info(self, case: ModuleCase) -> None:
        assert ModuleInfo.build(*case.info.parts) == case.info

    @inject_method()
    def test_from_qualname_str_is_same_info(self, case: ModuleCase) -> None:
        assert ModuleInfo.from_str(case.info.qualname) == case.info

    @inject_method()
    def test_from_python_module_is_same_info(self, case: ModuleCase) -> None:
        assert ModuleInfo.from_module(importlib.import_module(case.info.qualname)) == case.info


class TestTypeAnnotator:
    @inject_method()
    def test_annotate_is_valid(
        self,
        case: TypeCase,
        type_annotator: TypeAnnotator,
    ) -> None:
        assert type_annotator.annotate(case.info) == case.valid_annotation

    @inject_method()
    def test_parse_annotation_is_same_info(
        self,
        case: TypeCase,
        type_annotator: TypeAnnotator,
    ) -> None:
        assert type_annotator.parse(case.valid_annotation) == case.info


class TestTypeInspector:
    @inject_method()
    def test_inspect_python_type_is_same_info(
        self,
        case: TypeCase,
        type_inspector: TypeInspector,
    ) -> None:
        assert type_inspector.inspect(case.python_type) == case.info


class TestTypeLoader:
    @inject_method()
    def test_load_is_valid_python_type(
        self,
        case: TypeCase,
        type_loader: TypeLoader,
        type_inspector: TypeInspector,
    ) -> None:
        loaded = type_loader.load(case.info)

        assert loaded == case.python_type

    @inject_method()
    def test_load_origin_is_same_python_type_origin(
        self,
        case: TypeCase,
        type_loader: TypeLoader,
        type_inspector: TypeInspector,
    ) -> None:
        loaded = type_loader.load(case.info)

        assert t.get_origin(loaded) is t.get_origin(case.python_type)

    @inject_method()
    def test_load_args_are_same_python_type_args(
        self,
        case: TypeCase,
        type_loader: TypeLoader,
        type_inspector: TypeInspector,
    ) -> None:
        loaded = type_loader.load(case.info)

        assert t.get_args(loaded) == t.get_args(case.python_type)

    @pytest.mark.parametrize(
        ("info", "error"),
        [
            pytest.param(
                NamedTypeInfo("NonExistingType", builtins_module_info()),
                TypeLoaderError,
            ),
            pytest.param(
                NamedTypeInfo("SomeType", ModuleInfo("non_existing_module")),
                TypeLoaderError,
            ),
        ],
    )
    def test_load_raises_proper_error(self, type_loader: TypeLoader, info: TypeInfo, error: type[Exception]) -> None:
        with pytest.raises(error):
            type_loader.load(info)
