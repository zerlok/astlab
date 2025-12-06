from pytest_case_provider import inject_func

from tests.unit.case_builder import BuilderCase


@inject_func()
def test_module_build(case: BuilderCase) -> None:
    assert case.builder.render() == case.expected_code
