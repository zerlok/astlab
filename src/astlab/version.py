import enum
import sys
import typing as t

from astlab._typing import override


class PythonVersion(enum.Enum):
    PY39 = (3, 9)
    PY310 = (3, 10)
    PY311 = (3, 11)
    PY312 = (3, 12)
    PY313 = (3, 13)
    PY314 = (3, 14)

    @classmethod
    def get_current(cls) -> "PythonVersion":
        return cls(sys.version_info[:2])  # type: ignore[misc]

    @classmethod
    def get_all_supported(cls, *, ignore_outdated_runtime: bool = False) -> t.Iterable["PythonVersion"]:
        if ignore_outdated_runtime:
            return cls

        current = cls.get_current()
        return (v for v in cls if v <= current)

    @classmethod
    def parse(
        cls,
        value: t.Union["PythonVersion", t.Sequence[int], str, None],
        *,
        ignore_outdated_runtime: bool = False,
    ) -> "PythonVersion":
        if isinstance(value, PythonVersion):
            result = value

        elif isinstance(value, str):
            major, _, minor = value.partition(".")
            try:
                result = cls((int(major), int(minor)))

            except ValueError:
                msg = f"{value!r} is not a valid {cls.__name__}"
                raise ValueError(msg) from None

        elif isinstance(value, t.Sequence):
            result = cls(tuple(value[:2]))

        elif value is None:
            result = cls.get_current()

        else:
            t.assert_never(value)

        if not ignore_outdated_runtime and sys.version_info < result.value:
            msg = "current python runtime version is outdated to use the specified python version"
            raise ValueError(msg, sys.version_info, result)

        return result

    @override
    def __str__(self) -> str:
        major, minor = self.value
        return f"{major}.{minor}"

    @override
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self.name}>"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self.value < other.value

    def __le__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self.value <= other.value

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self.value > other.value

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self.value >= other.value
