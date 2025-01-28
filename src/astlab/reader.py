__all__ = [
    "import_module",
    "iter_package_modules",
    "parse_module",
    "walk_package_modules",
]


import ast
import importlib
import io
import sys
import typing as t
from pathlib import Path
from types import ModuleType


def walk_package_modules(root: Path) -> t.Iterable[Path]:
    for sub in root.rglob("*.py"):  # type: Path
        if sub.is_file():
            yield sub


def iter_package_modules(path: Path) -> t.Iterable[Path]:
    for sub in path.glob("*.py"):  # type: Path
        if sub.is_file():
            yield sub


def import_module(path: Path) -> ModuleType:
    relpath, src = min((path.relative_to(pypath), Path(pypath)) for pypath in sys.path if path.is_relative_to(pypath))
    qualname = ".".join((*relpath.parts[:-1], relpath.stem))
    return importlib.import_module(qualname)


def parse_module(source: t.Union[str, t.IO[str]], *, indented: bool = False) -> ast.Module:
    """Parse a block of code. The code may be indented, parser will shift the content to the left."""

    if not indented:
        return ast.parse(source if isinstance(source, str) else source.read())

    source = io.StringIO(source) if isinstance(source, str) else source
    offset: t.Optional[int] = None

    with io.StringIO() as dest:
        for line in source:
            if offset is None:
                idx, _ = next(((i, c) for i, c in enumerate(line) if not c.isspace()), (None, None))
                if idx is None:
                    continue

                offset = idx

            dest.writelines([line[offset:]])

        return ast.parse(dest.read())
