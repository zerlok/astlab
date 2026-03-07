import sys
import typing as t

import click

from astlab._typing import override
from astlab.version import PythonVersion


class PythonVersionParamType(click.ParamType):
    name = "python-version"

    def __init__(self, *, ignore_outdated_runtime: bool = False) -> None:
        self.__ignore_outdated_runtime = ignore_outdated_runtime

    @override
    def convert(
        self,
        value: t.Union[str, PythonVersion],
        param: t.Optional[click.Parameter],
        ctx: t.Optional[click.Context],
    ) -> PythonVersion:
        try:
            return PythonVersion.parse(value, ignore_outdated_runtime=self.__ignore_outdated_runtime)
        except ValueError:
            pass

        choices = ", ".join(str(v) for v in self.__iter_versions())
        self.fail(f"invalid python version {value!r}, expected one of: {choices}", param, ctx)

    if sys.version_info >= (3, 10):

        @override
        def get_metavar(self, param: click.Parameter, ctx: click.Context) -> str:  # noqa: ARG002
            return "|".join(str(v) for v in self.__iter_versions())

    else:

        @override
        def get_metavar(self, param: click.Parameter) -> str:  # noqa: ARG002
            return "|".join(str(v) for v in self.__iter_versions())

    def __iter_versions(self) -> t.Iterable[PythonVersion]:
        return PythonVersion.get_all_supported(ignore_outdated_runtime=self.__ignore_outdated_runtime)
