import typing as t

T = t.TypeVar("T")


class StubFoo:
    pass


class StubBar(t.Generic[T]):
    pass


class StubX:
    class Y:
        class Z:
            pass


StubInt = t.NewType("StubInt", int)
