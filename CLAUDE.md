# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install --all-extras

# Lint (check)
poetry run ruff check
poetry run ruff format --check

# Format (fix)
poetry run ruff check --fix
poetry run ruff format

# Type check
poetry run mypy

# Run tests (no coverage)
poetry run pytest --no-cov

# Run tests (with coverage)
poetry run pytest

# Run a single test
poetry run pytest tests/unit/test_builder.py::test_module_build

# Run a specific test case by name (pytest-case-provider parametrizes test names)
poetry run pytest tests/unit/test_builder.py -k "simple_module"

# Run all checks across all supported Python versions (3.9–3.14) via nox
nox -s ruff
nox -s mypy
nox -s pytest
```

## Architecture

**astlab** provides a fluent, context-manager-based API for constructing Python ASTs and rendering them to source code. The core pattern is using Python `with` statements to create nested scopes (modules, classes, methods), which matches the hierarchical structure of Python ASTs.

### Entry Points

`src/astlab/__init__.py` exposes two public functions:
- `astlab.module(name)` → `ModuleASTBuilder`
- `astlab.package(name)` → `PackageASTBuilder`

### Key Modules

**`src/astlab/builder.py`** — The main file containing all builder classes:
- `ModuleASTBuilder`: Top-level module builder; use `with` to enter scope, then call `.build()` or `.render()` or `.write()`
- `PackageASTBuilder`: Multi-module package builder; use `.module(name)` to access/create modules
- `ScopeASTBuilder`: Base class for any scope — provides expression builders (`attr`, `call`, `const`, `list_expr`, `union_type`, etc.) and statement builders (`assign_stmt`, `return_stmt`, `if_stmt`, `for_stmt`, `try_stmt`, `with_stmt`, etc.)
- `ClassScopeASTBuilder` / `ClassStatementASTBuilder`: Class definitions — supports `.dataclass()`, `.inherits()`, `.abstract()`, `.type_var()`, `.method_def()`, etc.
- `FuncStatementASTBuilder` / `MethodStatementASTBuilder`: Function/method definitions — supports `.arg()`, `.kwarg()`, `.returns()`, `.async_()`, `.abstract()`, `.stub()`, `.context_manager()`, etc.
- `TypeRefBuilder`: Wraps a `TypeInfo` and allows fluent type composition (`.optional()`, `.list()`, `.sequence()`, `.union()`, `.type_params()`, etc.)
- Expression builders: `AttrASTBuilder`, `CallASTBuilder`, `SliceASTBuilder` — support chaining via `.attr()`, `.call()`, `.index()`, `.slice()`, etc.

**`src/astlab/context.py`** — `BuildContext` holds shared mutable state during a build: current Python version, scope stack, module/package hierarchy, inter-module dependency tracking, and the resolver reference.

**`src/astlab/resolver.py`** — `DefaultASTResolver` converts `TypeExpr` / `TypeInfo` objects to `ast.expr` nodes. Handles automatic import dependency tracking, forward reference detection (wraps in string literals for recursive/self-referential types on Python < 3.14), and union type syntax variation.

**`src/astlab/types/`** — Type system:
- `model.py`: Immutable data models — `ModuleInfo`, `PackageInfo`, `NamedTypeInfo`, `TypeVarInfo`, `UnionTypeInfo`, `LiteralTypeInfo`, `EnumTypeInfo`
- `inspector.py`: `TypeInspector` — inspects runtime Python types (e.g., `int`, `dict[str, int]`, `typing.Optional[str]`) and converts them to `TypeInfo`
- `predef.py`: `get_predef()` — singleton providing pre-built `TypeInfo` objects for common types (`builtins.int`, `typing.Optional`, `typing.Generic`, `dataclasses.dataclass`, etc.)
- `annotator.py`: `TypeAnnotator` — converts `TypeInfo` to annotation strings
- `loader.py`: `ModuleLoader`, `TypeLoader` — for importing modules/types at runtime

**`src/astlab/version.py`** — `PythonVersion` enum (PY39–PY314) with comparison operators. Defaults to the current runtime's version. Controls syntax generation: union `|` syntax (≥3.10), `type X = Y` alias syntax (≥3.12), `class Foo[T]` type param syntax (≥3.12), forward reference elision (≥3.14).

**`src/astlab/writer.py`** — `render_module(ast.Module) → str` using `ast.unparse`, `write_module()` to write to file.

**`src/astlab/reader.py`** — `parse_module(code, indented=True) → ast.Module` for parsing source code (used in tests to normalize expected output).

### Testing Patterns

Tests use `pytest-case-provider`'s `@inject_func()` decorator. Test cases are plain functions in `tests/unit/case_builder.py` returning a `BuilderCase` dataclass. Each function becomes a separate parametrized test case. Version-specific tests are gated with `FeatureFlagMark` decorators from `tests/marks.py`:

```python
@FEATURE_UNION_TYPE_SYNTAX.mark_required()  # only runs on Python >= 3.10
def my_case() -> BuilderCase: ...

@FEATURE_TYPE_VAR_SYNTAX.mark_obsolete()    # only runs on Python < 3.12
def my_case() -> BuilderCase: ...
```

Expected code in tests is normalized via `normalize_code()` which round-trips through `ast.parse` → `ast.unparse` to eliminate formatting differences.

### Design Notes

- The `SIM117` ruff rule (merge nested `with` statements) is intentionally disabled because each `with` builder creates a distinct AST scope.
- All code uses strict mypy with `disallow_any_expr` and related settings. Type annotations must be complete and precise. `typing.Any` is only allowed in test fixtures.
- The library supports Python 3.9–3.14 and generates Python code targeting any of those versions — the `python_version` parameter at `build_module()`/`build_package()` controls which syntax is emitted.
- `ruff` ignores `ANN` (annotations) since mypy handles that, and `D` (docstrings) are not yet required.
