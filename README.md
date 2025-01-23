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
import astlab

# Create a new Python module
with astlab.module("foo") as foo:
    # Build a "Bar" dataclass
    with foo.class_def("Bar").dataclass() as bar:
        # Define a field "spam" of type int
        bar.field_def("spam", int)

# Generate and print the Python code from the AST
print(foo.render())
```

### Output:

```python
import builtins
import dataclasses

@dataclasses.dataclass()
class Bar:
    spam: builtins.int
```
