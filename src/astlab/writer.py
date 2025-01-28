from __future__ import annotations

__all__ = [
    "render_module",
    "write_module",
]

import ast
import typing as t

if t.TYPE_CHECKING:
    from pathlib import Path


def render_module(module: ast.Module) -> str:
    return ast.unparse(module)


def write_module(
    module: ast.Module,
    dest: Path,
    *,
    mode: t.Literal["w", "a"] = "w",
    mkdir: bool = False,
    exist_ok: bool = False,
) -> None:
    if dest.exists() and not exist_ok:
        msg = "module file already exists"
        raise ValueError(msg, dest)

    if mkdir:
        dest.parent.mkdir(parents=True, exist_ok=True)

    with dest.open(mode=mode) as out:
        out.write(render_module(module))
