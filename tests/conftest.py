import typing as t

import pytest
from _pytest.fixtures import SubRequest

from astlab.info import ModuleInfo, PackageInfo


@pytest.fixture
def stub_module_info(request: SubRequest) -> ModuleInfo:
    return ModuleInfo.from_module(request.module)


@pytest.fixture
def stub_package_info(stub_module_info: ModuleInfo) -> t.Optional[PackageInfo]:
    return stub_module_info.package
