"""Sphinx configuration: an intro page and API pages generated from the docstrings."""

project = "hybrid-bo"
copyright = "2025, Luis Kutschat"
author = "Luis Kutschat"

extensions = ["autoapi.extension"]

autoapi_dirs = ["../src/hybrid_bo"]
autoapi_member_order = "groupwise"
autoapi_own_page_level = "class"
autoapi_options = ["members", "undoc-members", "show-inheritance"]

exclude_patterns = ["_build"]
html_theme = "furo"
