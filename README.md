# htrflow_core

[![PyPI](https://img.shields.io/pypi/v/htrflow_core?style=flat-square)](https://pypi.python.org/pypi/htrflow_core/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/htrflow_core?style=flat-square)](https://pypi.python.org/pypi/htrflow_core/)
[![PyPI - License](https://img.shields.io/pypi/l/htrflow_core?style=flat-square)](https://pypi.python.org/pypi/htrflow_core/)

---

**Documentation**: [https://Swedish-National-Archives-AI-lab.github.io/htrflow_core](https://Swedish-National-Archives-AI-lab.github.io/htrflow_core)

**Source Code**: [https://github.com/Swedish-National-Archives-AI-lab/htrflow_core](https://github.com/Swedish-National-Archives-AI-lab/htrflow_core)

**PyPI**: [https://pypi.org/project/htrflow_core/](https://pypi.org/project/htrflow_core/)

---

A short description of the project

## Installation

```sh
pip install htrflow_core
```

## Development

- Clone this repository
- Requirements:
  - [Poetry](https://python-poetry.org/)
  - Python 3.10+
- Create a virtual environment and install the dependencies

```sh
poetry install
```

- Activate the virtual environment

```sh
poetry shell
```

### Testing

Poetry provides a run command to execute the given command inside the project’s virtual environment. So execute the following command to run the tests:

```sh
poetry run pytest -v
```

And if your inside the virtual environment run:

```sh
pytest
```

---

### Building the package

For Poetry, the equivalent of `pip install -e` . (which is used to install a package in editable mode with pip) is to use the poetry install command in the root directory of the project.

When you run `poetry install` in a project that is managed by Poetry, it installs the project's dependencies as well as the project itself in editable mode. This means that changes to the project's code will immediately affect the installed package without the need for reinstallation.

> You can run poetry install whether you're inside the Poetry-created virtual environment (activated using poetry shell) or not. Poetry will handle the installation of the project in editable mode correctly in either case.

---

### Publish the Package

But before you can publish your library, you need to package it with the build command:

```sh
poetry build
```

To publish your library, you will need to properly configure your PyPI credentials, as Poetry will publish the library to PyPI by default.

```sh
poetry config repositories.test-pypi https://test.pypi.org/legacy/
```

Configure key:

```sh
poetry config pypi-token.test-pypi pypi-<tokenv_value>
```

Once the library is packaged, you can use the publish command to publish it.

```sh
# Only `poetry publish` will push by default to pypi directly
poetry publish -r test-pypi
```
