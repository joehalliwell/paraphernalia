test:
    pytest -v --log-cli-level=INFO

makedocs:
    cd docs && sphinx-apidoc -f -o source ../paraphernalia
    cd docs && make html
