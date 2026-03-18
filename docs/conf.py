# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "hybrid-bo"
copyright = "2025, Luis Kutschat"
author = "Luis Kutschat"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "autoapi.extension"]

autoapi_dirs = ["../src/hybrid_bo"]
autoapi_member_order = "groupwise"
autoapi_own_page_level = "class"

autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "imported-members",
]

autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
