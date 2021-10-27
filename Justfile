test:
    coverage run --source=paraphernalia -m pytest -v -rs --log-cli-level=INFO
    coverage report

makedocs:
    cd docs && sphinx-apidoc --module-first --separate --force -o source ../paraphernalia
    cd docs && make clean html
