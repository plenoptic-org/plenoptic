# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import glob
import os
import pathlib
import sys
from importlib.metadata import version

import torch

# by default, torch uses all avail threads which slows things run in parallel
torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("./tutorials/"))


# -- Project information -----------------------------------------------------

project = "plenoptic"
copyright = "2019-2025, Plenoptic authors"
author = "Plenoptic authors"

release: str = version("plenoptic")
# this will grab major.minor.patch (excluding any .devN afterwards, which should only
# show up when building locally during development)
version: str = ".".join(release.split(".")[:3])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "matplotlib.sphinxext.plot_directive",
    "matplotlib.sphinxext.mathmpl",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "myst_nb",
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "sphinx.ext.viewcode",
]

numfig = True
add_module_names = False

nitpicky = True

intersphinx_mapping = {
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyrtools": ("https://pyrtools.readthedocs.io/en/latest", None),
    "python": ("https://docs.python.org/3/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    ".jupyter_cache",
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
# when napoleon_use_rtype is true, the return type is often confused. setting this to
# false let's sphinx-autodoc-typehints handle it instead.
napoleon_use_rtype = False

# SPHINX AUTODOC TYPEHINTS

always_use_bars_union = True
typehints_defaults = "braces"

# SPHINX CROSS REFERENCES

add_function_parentheses = False

# AUTOSUMMARY / AUTODOC

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "groupwise",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Path for static files (custom stylesheets or JavaScript)
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "plenopticdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "plenoptic.tex",
        "plenoptic Documentation",
        "Plenoptic authors",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "plenoptic", "plenoptic Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "plenoptic",
        "plenoptic Documentation",
        author,
        "plenoptic",
        "Visual Information Processing",
        "Miscellaneous",
    ),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------

# MATPLOTLIB
# because of the examples in the docstrings, want to default to showing source and not
# showing link. in actual doc pages, this needs to be reversed.
plot_include_source = True
plot_html_show_source_link = False

# MYST

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "attrs_block",
]

# SPHINXCONTRIB-BIBTEX
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# SPHINX COPYBUTTON

# skip prompt characters and console outputs when copying
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#automatic-exclusion-of-prompts-from-the-copies
copybutton_exclude = ".linenos, .gp, .go"

# MYST_NB

# max time (in secs) per notebook cell. here, we disable this
nb_execution_timeout = -1
# by default, we don't run any of the notebooks (they take too long). users can override
# this behavior using env variables
if run_nb := os.environ.get("RUN_NB"):
    if run_nb == "1" or run_nb == "all":
        nb_execution_excludepatterns = []
        print("Running all notebooks, things will take longer...")
    else:
        all_nbs = glob.glob("tutorials/**/*md", recursive=True)
        all_nbs = [pathlib.Path(n).stem for n in all_nbs]
        run_globs = [f"*{n}*" for n in run_nb.split(",")]
        nb_execution_excludepatterns = [
            f"*{n}*"
            for n in all_nbs
            if not any([glob.fnmatch.fnmatch(n, g) for g in run_globs])
        ]
        print(f"Excluding notebooks: {nb_execution_excludepatterns}")
else:
    nb_execution_excludepatterns = ["*"]
    print("Not running any notebooks, see CONTRIBUTING for details")

nb_execution_mode = os.environ.get("NB_EXECUTION_MODE", "cache")
nb_execution_raise_on_error = True
nb_execution_cache_path = ".jupyter_cache"
