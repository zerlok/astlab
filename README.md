# astlab

[![Latest Version](https://img.shields.io/pypi/v/astlab.svg)](https://pypi.python.org/pypi/astlab)
[![Python Supported Versions](https://img.shields.io/pypi/pyversions/astlab.svg)](https://pypi.python.org/pypi/astlab)
[![MyPy Strict](https://img.shields.io/badge/mypy-strict-blue)](https://mypy.readthedocs.io/en/stable/getting_started.html#strict-mode-and-configuration)
[![Test Coverage](https://codecov.io/gh/zerlok/astlab/branch/main/graph/badge.svg)](https://codecov.io/gh/zerlok/astlab)
[![Downloads](https://img.shields.io/pypi/dm/astlab.svg)](https://pypistats.org/packages/astlab)
[![GitHub stars](https://img.shields.io/github/stars/zerlok/astlab)](https://github.com/zerlok/astlab/stargazers)

**astlab** is a Python library that provides an intuitive API for building and manipulating Abstract Syntax Trees (ASTs) to generate Python code. With **astlab**, you can easily construct Python modules, classes, functions, type aliases, and generics using a fluent API — then render them into valid, executable Python code.

## Features

* **Easy AST construction**: Build Python code using a fluent, structured API.
* **Code generation**: Generate fully valid, formatted Python source without templates.
* **Supports nested scopes & auto imports**: Create classes, methods, and nested modules with automatic import resolution.
* **Type system support**: Define and use **type variables**, **generic classes**, and **type aliases** compatible with Python 3.9–3.14 syntax.
* **Highly customizable**: Extend the builder model for any Python AST use case.

## Installation

```bash
pip install astlab
```

## Usage

### Simple Example

```python
import ast
import astlab

with astlab.module("foo") as foo:
    with foo.class_def("Bar").dataclass() as bar:
        bar.field_def("spam", int)

print(foo.render())
print(ast.dump(foo.build(), indent=4))
```

#### Output

```python
import builtins
import dataclasses

@dataclasses.dataclass()
class Bar:
    spam: builtins.int
```

---

### Function Definition & Call Example

```python
import astlab

with astlab.module("foo") as foo:
    with foo.class_def("Bar") as bar:
        with bar.method_def("do_stuff").arg("spam", int).returns(str) as stuff:
            stuff.assign_stmt("result", stuff.call(str).arg(stuff.attr("spam")))
            stuff.return_stmt(stuff.attr("result"))

print(foo.render())
```

#### Output

```python
import builtins

class Bar:

    def do_stuff(self, spam: builtins.int) -> builtins.str:
        result = builtins.str(spam)
        return result
```

---

### Type Reference Example

```python
import astlab

with astlab.package("main") as main:
    with main.module("foo") as foo:
        with foo.class_def("Bar") as bar:
            pass

    with main.module("spam") as spam:
        with spam.class_def("Eggs").inherits(bar) as eggs:
            with eggs.method_def("do_stuff").returns(bar.ref().optional()) as stuff:
                pass

print(spam.render())
```

#### Output

```python
import main.foo
import typing

class Eggs(main.foo.Bar):

    def do_stuff(self) -> typing.Optional[main.foo.Bar]:
        pass
```

---

### Generics and Type Variables

**astlab** supports defining type variables and generic classes.
Both the legacy (`typing.TypeVar`) and modern (`class Node[T: int]`) syntaxes are supported depending on Python version.

#### Example

```python
import astlab

with astlab.module("generic") as mod:
    with mod.class_def("Node") as node, node.type_var("T").lower(int) as T:
        node.field_def("value", T)
        node.field_def("parent", node.ref().type_params(type_var).optional(), mod.none())

print(mod.render())
```

#### Output (python < 3.12)

```python
import builtins
import typing

T = typing.TypeVar('T', bound=builtins.int)

class Node(typing.Generic[T]):
    value: T
    parent: typing.Optional['Node[T]'] = None
```

#### Output (python 3.12, 3.13)

```python
import builtins
import typing

class Node[T: builtins.int]:
    value: T
    parent: typing.Optional['Node[T]'] = None
```

#### Output (python ≥ 3.14)

```python
import builtins
import typing

class Node[T: builtins.int]:
    value: T
    parent: typing.Optional[Node[T]] = None
```

---

### Type Aliases

**astlab** allows declarative creation of type aliases, including recursive and generic aliases.
It automatically emits valid syntax for both `typing.TypeAlias` (pre-3.12) and `type X = Y` (3.12+).

#### Example

```python
import astlab
from astlab.types import predef

with astlab.module("alias") as mod:
    mod.type_alias("MyInt").assign(int)

    with mod.type_alias("Json") as json_alias:
        json_alias.assign(
            json_alias.union_type(
                None,
                bool,
                int,
                float,
                str,
                mod.list_type(json_alias),
                mod.dict_type(str, json_alias),
            )
        )

    with (
        mod.type_alias("Nested") as nested_alias,
        nested_alias.type_var("T") as T,
    ):
        nested_alias.assign(
            nested_alias.union_type(
                T
                nested_alias.sequence_type(nested_alias.type_params(T)),
            )
        )
```

#### Output (python < 3.12)

```python
import builtins
import typing

MyInt: typing.TypeAlias = builtins.int
Json: typing.TypeAlias = typing.Union[
    None,
    builtins.bool,
    builtins.int,
    builtins.float,
    builtins.str,
    builtins.list['Json'],
    builtins.dict[builtins.str, 'Json'],
]
T = typing.TypeVar("T")
Nested: typing.TypeAlias = typing.Union[T, typing.Sequence['Nested[T]']]
```

#### Output (python 3.12, 3.13)

```python
import builtins
import typing

type MyInt = builtins.int
type Json = typing.Union[
    None,
    builtins.bool,
    builtins.int,
    builtins.float,
    builtins.str,
    builtins.list['Json'],
    builtins.dict[builtins.str, 'Json'],
]
type Nested[T] = typing.Union[T, typing.Sequence['Nested[T]']]
```

#### Output (python ≥ 3.14)

```python
import builtins
import typing

type MyInt = builtins.int
type Json = typing.Union[
    None,
    builtins.bool,
    builtins.int,
    builtins.float,
    builtins.str,
    builtins.list[Json],
    builtins.dict[builtins.str, Json],
]
type Nested[T] = typing.Union[T, typing.Sequence[Nested[T]]]
```