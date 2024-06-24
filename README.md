# OrcaDemos

## Setup

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
