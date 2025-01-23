from __future__ import annotations

__all__ = [
    "BuildContext",
    "Scope",
]


import typing as t
from dataclasses import dataclass

if t.TYPE_CHECKING:
    import ast

    from astlab._typing import Self
    from astlab.info import ModuleInfo, PackageInfo


@dataclass()
class Scope:
    name: t.Optional[str]
    body: list[ast.stmt]


class BuildContext:
    def __init__(
        self,
        packages: t.MutableSequence[PackageInfo],
        dependencies: t.MutableMapping[ModuleInfo, t.MutableSet[ModuleInfo]],
        scopes: t.MutableSequence[Scope],
    ) -> None:
        self.__packages = packages
        self.__dependencies = dependencies
        self.__scopes = scopes
        self.__module: t.Optional[ModuleInfo] = None

    @property
    def module(self) -> ModuleInfo:
        assert self.__module is not None
        return self.__module

    def enter_package(self, info: PackageInfo) -> Self:
        assert self.__module is None
        self.__packages.append(info)
        return self

    def leave_package(self) -> Self:
        assert self.__module is None
        self.__packages.pop()
        return self

    def enter_module(self, info: ModuleInfo, body: list[ast.stmt]) -> Scope:
        assert self.__module is None
        assert len(self.__scopes) == 0
        self.__module = info
        return self.enter_scope(None, body)

    def leave_module(self) -> Scope:
        assert len(self.__scopes) == 1
        scope = self.leave_scope()
        self.__module = None
        return scope

    def enter_scope(self, name: t.Optional[str], body: list[ast.stmt]) -> Scope:
        scope = Scope(name, body)
        self.__scopes.append(scope)
        return scope

    def leave_scope(self) -> Scope:
        return self.__scopes.pop()

    @property
    def namespace(self) -> t.Sequence[str]:
        return tuple(scope.name for scope in self.__scopes if scope.name is not None)

    @property
    def name(self) -> str:
        return self.namespace[-1]

    @property
    def current_dependencies(self) -> t.MutableSet[ModuleInfo]:
        return self.__dependencies[self.module]

    def get_dependencies(self, info: ModuleInfo) -> t.Collection[ModuleInfo]:
        return self.__dependencies[info]

    @property
    def current_scope(self) -> Scope:
        return self.__scopes[-1]

    @property
    def current_body(self) -> list[ast.stmt]:
        return self.current_scope.body

    def append_body(self, stmt: t.Optional[ast.stmt]) -> None:
        if stmt is not None:
            self.current_body.append(stmt)

    def extend_body(self, stmts: t.Sequence[t.Optional[ast.stmt]]) -> None:
        self.current_body.extend(stmt for stmt in stmts if stmt is not None)
