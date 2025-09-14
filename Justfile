test:
    uv run coverage run --source src --module pytest tests/ -v -ra --log-cli-level=INFO
    uv run coverage report -m

publish:
    rm -rf dist
    uv build
    uv publish

sync:
    uv sync --all-extras

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
