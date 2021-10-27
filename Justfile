test:
    coverage run --source=paraphernalia -m pytest -v -rs --log-cli-level=INFO
    coverage report

makedocs:
    cd docs && rm -rf source/generated && sphinx-apidoc --module-first --no-toc --templatedir=source/_templates --separate --force -o source/generated ../paraphernalia
    cd docs && make clean html
