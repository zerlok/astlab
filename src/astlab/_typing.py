__all__ = [
    "EllipsisType",
    "Self",
    "TypeAlias",
    "TypeGuard",
    "assert_never",
    "override",
]

import typing as t

# NOTE: this allows to annotate methods with typing extensions during runtime (when typing_extensions is not installed).
if t.TYPE_CHECKING:
    import sys

    from typing_extensions import Self, TypeAlias, TypeGuard, assert_never, override

    if sys.version_info < (3, 10):
        EllipsisType = type(Ellipsis)

    else:
        from types import EllipsisType

else:
    Self = t.Any
    TypeGuard = t.Optional
    TypeAlias = t.Any
    EllipsisType = t.Any

    def assert_never(*args: object) -> t.NoReturn:
        msg = "Expected code to be unreachable"
        raise AssertionError(msg, *args)

    def override(func: object) -> object:
        return func
