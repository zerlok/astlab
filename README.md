# astlab

[![Latest Version](https://img.shields.io/pypi/v/astlab.svg)](https://pypi.python.org/pypi/astlab)
[![Python Supported Versions](https://img.shields.io/pypi/pyversions/astlab.svg)](https://pypi.python.org/pypi/astlab)
[![MyPy Strict](https://img.shields.io/badge/mypy-strict-blue)](https://mypy.readthedocs.io/en/stable/getting_started.html#strict-mode-and-configuration)
[![Test Coverage](https://codecov.io/gh/zerlok/astlab/branch/main/graph/badge.svg)](https://codecov.io/gh/zerlok/astlab)
[![Downloads](https://img.shields.io/pypi/dm/astlab.svg)](https://pypistats.org/packages/astlab)
[![GitHub stars](https://img.shields.io/github/stars/zerlok/astlab)](https://github.com/zerlok/astlab/stargazers)

**astlab** is a Python library that provides an intuitive API for building and manipulating Abstract Syntax Trees (ASTs) to generate Python code. With **astlab**, you can easily create Python modules, classes, fields, and more using a simple and readable syntax, and convert the AST back into executable Python code.

## Features

- **Easy AST construction**: Build Python code using a fluent and intuitive API.
- **Code generation**: Convert your AST into valid Python code, forget about jinja templates.
- **Supports nested scopes & auto imports**: Create nested classes, methods, and fields effortlessly. Reference types from other modules easily.
- **Highly customizable**: Extend and modify the API to suit your needs.

## Installation

You can install **astlab** from PyPI using `pip`:

```bash
pip install astlab
```

## Usage

### Simple example

Here's a basic example of how to use **astlab** to create a Python module with a dataclass.

```python
import ast
import astlab

# Create a new Python module
with astlab.module("foo") as foo:
    # Build a "Bar" dataclass
    with foo.class_def("Bar").dataclass() as bar:
        # Define a field "spam" of type int
        bar.field_def("spam", int)

# Generate and print the Python code from the AST
print(foo.render())
# Or you can just get the AST
print(ast.dump(foo.build(), indent=4))
```

#### Output

Render:

```python
import builtins
import dataclasses

@dataclasses.dataclass()
class Bar:
    spam: builtins.int
```

Dump built AST:

```python
Module(
    body=[
        Import(
            names=[
                alias(name='builtins')]),
        Import(
            names=[
                alias(name='dataclasses')]),
        ClassDef(
            name='Bar',
            bases=[],
            keywords=[],
            body=[
                AnnAssign(
                    target=Name(id='spam'),
                    annotation=Attribute(
                        value=Name(id='builtins'),
                        attr='int'),
                    simple=1)],
            decorator_list=[
                Call(
                    func=Attribute(
                        value=Name(id='dataclasses'),
                        attr='dataclass'),
                    args=[],
                    keywords=[])])],
    type_ignores=[])
```

### Func def & call example

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

### Type reference example

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