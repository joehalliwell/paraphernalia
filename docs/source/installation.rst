Installation
============

At the moment, it's best to install directly from github as follows:

.. code-block:: bash

    %%bash
    if ! $(python -c "import paraphernalia" &> /dev/null); then
        pip install git+https://github.com/joehalliwell/paraphernalia.git#egg=paraphernalia[openai,taming]
        pip uninstall --yes torchtext
    fi
    # Refresh
    pip install --upgrade --force-reinstall --no-deps git+https://github.com/joehalliwell/paraphernalia.git#egg=paraphernalia[openai,taming]

Optional dependencies
---------------------

- openai: CLIP and associated tools from OpenAI
- taming: image generators from Taming Transformers
