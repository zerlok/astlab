import pytest
from _pytest.fixtures import SubRequest

from astlab.info import ModuleInfo, PackageInfo


@pytest.fixture
def stub_package_info(request: SubRequest) -> PackageInfo:
    mod = ModuleInfo.from_module(request.module)
    return PackageInfo(mod.package, mod.name)
