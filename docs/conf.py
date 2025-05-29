# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../src/"))

project = "SigKit"
copyright = "2025, Isaiah Harville, Joshua Payne"
author = "Isaiah Harville, Joshua Payne"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # auto‐generate docs from docstrings
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",  # add links to source code
    "sphinx.ext.autosummary",  # one page per module
]

autosummary_generate = True
autosummary_imported_members = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_theme = "piccolo_theme"
html_theme_options = {
    "logo": {"text": "SigKit"},
    "show_prev_next": False,
    "navigation_depth": 2,
}
