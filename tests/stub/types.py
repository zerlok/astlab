import sys
import typing as t
from dataclasses import dataclass
from typing import TypeAlias

from astlab._typing import override

T = t.TypeVar("T")


class StubFoo:
    pass


class StubBar(t.Generic[T]):
    pass


class StubX:
    class Y:
        class Z:
            pass


class StubCM(t.ContextManager["StubCM"]):
    @override
    def __exit__(self, exc_type: object, exc_value: object, traceback: object, /) -> None:
        pass


@dataclass(frozen=True)
class StubNode(t.Generic[T]):
    value: T
    parent: t.Optional["StubNode[T]"] = None


StubInt = t.NewType("StubInt", int)

StubUnionAlias: TypeAlias = t.Union[StubFoo, StubBar[StubInt], StubX]

if sys.version_info >= (3, 12):
    type StubNumber = int | float

else:
    StubNumber = ...
