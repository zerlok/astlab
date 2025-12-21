import enum
import sys
import typing as t


class PythonVersion(enum.Enum):
    PY39 = (3, 9)
    PY310 = (3, 10)
    PY311 = (3, 11)
    PY312 = (3, 12)
    PY313 = (3, 13)
    PY314 = (3, 14)

    @classmethod
    def get(cls, value: t.Union["PythonVersion", t.Sequence[int], None] = None) -> "PythonVersion":
        if isinstance(value, PythonVersion):
            return value

        else:
            target = tuple(value[:2]) if value is not None else sys.version_info[:2]  # type: ignore[misc]

            cls.__validate(target)

            return cls(target) if target <= max(cls).value else max(cls)

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

    @classmethod
    def __validate(cls, version: tuple[int, ...]) -> None:
        if sys.version_info < version:
            msg = "the current version of python is too old"
            raise RuntimeError(msg, sys.version_info, version)

        if version < min(cls).value:
            msg = "the specified version of python is too old"
            raise RuntimeError(msg, version)
