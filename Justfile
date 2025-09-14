# Run tests with coverage and report
test:
    uv run coverage run --source src --module pytest tests/ -v -ra --log-cli-level=INFO
    uv run coverage report -m

# Format and fix
format:
    ruff check --select I --fix .
    ruff format .

# Publish the package to PyPI
publish:
    rm -rf dist
    uv build
    uv publish

# Synch the venv
sync:
    uv sync --all-extras

# Regenerate doc stubs from the packages and build the docs
makedocs:
    #!/usr/bin/env bash
    set -euxo pipefail
    cd docs
    rm -rf source/generated
    SPHINX_APIDOC_OPTIONS="members,undoc-members" sphinx-apidoc \
        --module-first --no-toc --separate \
        --templatedir=source/_templates -o source/generated \
        ../src/paraphernalia
    make clean html
