# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata as metadata
import logging
import os
import sys
from pathlib import Path

project = "noob"
copyright = "2025, raymond, jonny"
author = "raymond, jonny"
release = metadata.version("noob")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.append(str((Path(__file__).parent / "_ext").resolve()))

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "myst_nb",
    "sphinx.ext.todo",
    "plot",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "notes.md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    # make myst-nb code blocks not look like shit
    "css/notebooks.css",
    "css/noob-js.css",
]
html_js_files = ["js/noob-js.js"]

# --------------------------------------------------
# Extension options
# --------------------------------------------------

# Autodoc
autoclass_content = "class"
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"
add_module_names = False

# autodoc-pydantic
autodoc_pydantic_model_show_json_error_strategy = "coerce"
autodoc_pydantic_model_show_json = True

# napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# todo
todo_include_todos = True
todo_link_only = True

# myst-nb
nb_render_markdown_format = "myst"
nb_execution_show_tb = True

# inheritance-diagram
inheritance_graph_attrs = {"rankdir": "LR", "splines": "ortho"}

inheritance_edge_attrs = {
    "color": "blue",
    "style": "bold",
}


class FuckTheSphinxFiltersFilter(logging.Filter):
    """
    A filter that goes like "fuck the sphinx logging filters that ignores our warning filters"

    Use this whenever there are warnings that cause CI to fail but you can't actually
    do normal python things to suppress the warnings because
    """

    def filter(self, record: logging.LogRecord):
        # filter warnings that are NOT OUR FAULT
        if hasattr(record, "location") and record.location is not None:
            if "typing.Annotated" in record.location or "typing.Union" in record.location:
                return False

        # not worth installing graphviz for one diagram in gh actions testing
        if (
            "GITHUB_ACTION" in os.environ
            and hasattr(record, "getMessage")
            and "dot command" in record.getMessage()
        ):
            return False

        return True


def setup(app):
    # rm this once #79 is merged
    from noob.config import config

    config.user_dir = Path(__file__).parent / "assets" / "pipelines"

    logger = logging.getLogger("sphinx")
    logger.filters.insert(0, FuckTheSphinxFiltersFilter())
