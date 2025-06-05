import typing as t

import pytest
from _pytest.fixtures import SubRequest

from astlab.types.annotator import TypeAnnotator
from astlab.types.inspector import TypeInspector
from astlab.types.loader import ModuleLoader, TypeLoader
from astlab.types.model import ModuleInfo, PackageInfo


@pytest.fixture
def stub_module_info(request: SubRequest) -> ModuleInfo:
    return ModuleInfo.build(*request.module.__name__.split("."))


@pytest.fixture
def stub_package_info(stub_module_info: ModuleInfo) -> t.Optional[PackageInfo]:
    return stub_module_info.package


@pytest.fixture
def module_loader() -> ModuleLoader:
    return ModuleLoader()


@pytest.fixture
def type_loader() -> TypeLoader:
    return TypeLoader()


@pytest.fixture
def type_annotator(module_loader: ModuleLoader) -> TypeAnnotator:
    return TypeAnnotator(module_loader)


@pytest.fixture
def type_inspector() -> TypeInspector:
    return TypeInspector()
