# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "paraphernalia"
copyright = "2021, Joe Halliwell"
author = "Joe Halliwell"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx_click",  # Generate docs for click programs
    "sphinxcontrib.autodoc_pydantic",  # Nicer docs for pydantic models
]

napoleon_include_init_with_doc = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for autodoc -----------------------------------------------------

# Sort functions etc. by order of appearance in the source
autodoc_member_order = "bysource"

# Use both class and __init__ docs
autoclass_content = "both"

# Display the __init__ signature with the class
autodoc_class_signature = "mixed"

# Show typehints in signature not online
autodoc_typehints = "description"

# Turn on sphinx.ext.autosummary
autosummary_generate = True

add_module_names = False  # Turn off full qualification

modindex_common_prefix = ["paraphernalia."]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "paraphernalia.css",
]

html_theme_options = {
    "show_toc_level": 2,
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/joehalliwell/paraphernalia",
            "icon": "fab fa-github-square",
        },
    ],
}

# For "Edit this Page" button
html_context = {
    "github_user": "joehalliwell",
    "github_repo": "paraphernalia",
    "github_version": "main",
    "doc_path": "docs/source",
}


autodoc_pydantic_model_show_json = True
autodoc_pydantic_settings_show_json = False
