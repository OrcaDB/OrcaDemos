# OrcaDemos

## Setup

### Install asdf

We recommend using [asdf](https://asdf-vm.com) to manage your Python, Node, and Poetry versions. Please install asdf using the following instructions:

From any folder

- `brew install asdf`
- `asdf plugin add python`
- `asdf plugin add nodejs`
- `asdf plugin add poetry`

From the OrcaDemos root folder, run the following command to install all proper tool versions

- `asdf install`

**Troubleshooting**

If you see the below error

```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/stephaniehabib/.asdf/installs/python/3.11.5/lib/python3.11/lzma.py", line 27, in <module>
    from _lzma import *
ModuleNotFoundError: No module named '_lzma'
WARNING: The Python lzma extension was not compiled. Missing the lzma lib?
```

Run `brew install xz`

If using zsh please update your .zshrc file

Run `echo -e "\n. $(brew --prefix asdf)/libexec/asdf.sh" >> ${ZDOTDIR:-~}/.zshrc`

https://asdf-vm.com/guide/getting-started.html

### Install Pre-Commit Hooks

We use pre-commit hooks to sanitize code. Please install pre-commit hooks using the following instructions:

- `pip install pre-commit` or `brew install pre-commit`

  Troubleshooting Note: If you install via homebrew, you may get an error like `RuntimeError: failed to find interpreter for Builtin discover of python_spec='python3.11'`. In this case, you will need to brew install the python version specified as the black `language_version` in `.pre-commit-config.yaml`

- `pre-commit install`

Run `poetry install` to install the dependencies and create a virtual environment.

The lock file isn't checked in because we need to specify different Torch versions for different platforms, and
[there seem to be issues with generating the Poetry Lock file with matchers and specifies paths/URLs.](https://github.com/python-poetry/poetry/issues/3959#issuecomment-1251672579)

## Use

Ensure that an Orca DB server is running (follow instructions in the  [`manifold` README](https://github.com/OrcaDB/orca/blob/main/README.md)). Once the server is running, `orcalib` will automatically find the local server.

To run a script simply do `poetry run python scripts/<script_name>.py`.

To run a notebook select the correct `./.venv/bin/python` kernel.

## Todos
- Cleanup notebooks
- Generate canonical datasets
