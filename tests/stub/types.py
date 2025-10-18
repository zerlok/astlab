import enum
import sys
import typing as t
from dataclasses import dataclass

from typing_extensions import TypeAlias

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


class StubEnum(enum.Enum):
    FOO = enum.auto()
    BAR = enum.auto()


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

StubRecursive: TypeAlias = t.Union[T, t.Sequence["StubRecursive[T]"]]

if sys.version_info >= (3, 10):
    StubUnionType = int | str | float | None

else:
    StubUnionType = t.Union[int, str, float, None]

# TODO: enable after python 3.9, 3.10, 3.11 version support stop drop.
#   type StubRecursive[T] = T | t.Sequence[StubRecursive[T]]  # noqa: ERA001
