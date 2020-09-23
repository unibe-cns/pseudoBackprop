# pseudoBackprop
Project to compare vanilla backprop, feedback alignment and pseudo backpropagation on a simple example

## Maintenance manual

Notes on how to build and contribute to the code

### Installation

The package is currently only available for local installation using setuptools.
To install the local package type in the root of the project:
```bash
pip install .
```

### Testing

We use [nosetest](https://nose.readthedocs.io/en/latest/) fro testing.
To run all the tests type in the root of the project directory:
```bash
nosetests
```
Or to run a single test write for example
```bash
nosetest tests/import_test.py
```

### Codestyle and code-hygiene

To ensure style and quality run [pycodestyle](https://pypi.org/project/pycodestyle/) and [pylint](http://pylint.pycqa.org/en/latest/) on the source code.
For pycodestyle run for example:
```bash
pycodestyle tests/import_test.py
```
and for pylint:
```bash
pylint tests/import_test.py
```
