test:
    coverage run --source=paraphernalia -m pytest -v --log-cli-level=INFO
    coverage report

makedocs:
    cd docs && sphinx-apidoc --module-first -f -o source ../paraphernalia
    cd docs && make html
