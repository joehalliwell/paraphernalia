test:
    coverage run --source=paraphernalia -m pytest -v -rs --log-cli-level=INFO
    coverage report

makedocs:
    #!/usr/bin/env bash
    set -euxo pipefail
    cd docs
    rm -rf source/generated
    SPHINX_APIDOC_OPTIONS="members,undoc-members" sphinx-apidoc \
        --module-first --no-toc --separate \
        --templatedir=source/_templates -o source/generated \
        ../paraphernalia
    make clean html
