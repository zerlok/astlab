import typing as t

import click
import pytest
from click.testing import CliRunner

from astlab.click import PythonVersionParamType
from astlab.version import PythonVersion


class TestPythonVersionParamType:
    def test_help_renders_metavar_properly(
        self,
        cli: CliRunner,
        stuff: click.Command,
        versions: t.Sequence[PythonVersion],
    ) -> None:
        result = cli.invoke(stuff, ["--help"])
        expected_metavar = "|".join(str(v) for v in PythonVersion.get_all_supported())
        assert f"-p, --python {expected_metavar}" in result.output

    def test_default_value_set_properly(
        self,
        cli: CliRunner,
        stuff: click.Command,
        versions: t.Sequence[PythonVersion],
    ) -> None:
        cli.invoke(stuff)
        assert versions == [PythonVersion.get_current()]

    @pytest.mark.parametrize("ver", list(PythonVersion.get_all_supported()))
    def test_version_parsed_properly(
        self,
        cli: CliRunner,
        stuff: click.Command,
        versions: t.Sequence[PythonVersion],
        ver: str,
    ) -> None:
        cli.invoke(stuff, ["--python", ver])
        assert versions == [PythonVersion.parse(ver)]

    @pytest.mark.parametrize("ver", ["invalid", "2.7", "3.3", "6.8"])
    def test_invalid_version_raises_error(
        self,
        cli: CliRunner,
        stuff: click.Command,
        ver: str,
    ) -> None:
        result = cli.invoke(stuff, ["--python", ver])
        assert f"Invalid value for '-p' / '--python': invalid python version '{ver}'" in result.output

    @pytest.fixture
    def stuff(self, versions: list[PythonVersion]) -> click.Command:
        @click.command("stuff")
        @click.option(
            "-p",
            "--python",
            "python_version",
            type=PythonVersionParamType(),
            default=PythonVersion.get_current(),
        )
        def do_stuff(
            python_version: PythonVersion,
        ) -> None:
            versions.append(python_version)

        return do_stuff

    @pytest.fixture
    def versions(self) -> list[PythonVersion]:
        return []


@pytest.fixture
def cli() -> CliRunner:
    return CliRunner(catch_exceptions=False)
