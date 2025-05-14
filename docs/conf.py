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
import inspect
import os
import sys
from importlib.metadata import version

import torch

import plenoptic

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("./tutorials/"))


# -- Project information -----------------------------------------------------

project = "plenoptic"
copyright = "2019, Plenoptic authors"
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
    "numpydoc",
    "nbsphinx",
    "nbsphinx_link",
    "matplotlib.sphinxext.plot_directive",
    "matplotlib.sphinxext.mathmpl",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.apidoc",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "sphinx.ext.viewcode",
]

add_module_names = False

intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

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
napoleon_use_rtype = False

numfig = True

# numpydoc


# find whether the object is part of plenoptic
def is_part_of_plenoptic(obj):
    try:
        return obj.__module__.startswith("plenoptic")
    except AttributeError:
        # then it's a module
        return obj.__name__.startswith("plenoptic")


def is_interesting(obj):
    # find whether the object is a class or a module
    is_right_object = inspect.isclass(obj) or inspect.ismodule(obj)
    return is_right_object and is_part_of_plenoptic(obj)


def find_module(obj):
    members = inspect.getmembers(obj, is_interesting)
    to_return = []
    for name, mem in set(members):
        if inspect.ismodule(mem):
            # go through the modules recursively
            to_return.extend(find_module(mem))
        else:
            # if they inherit torch Module
            if issubclass(mem, torch.nn.Module):
                to_return.append(f"{mem.__module__}.{name}")
    # remove duplicates
    return set(to_return)


# avoid showing all the torch.nn.Module attributes and methods
numpydoc_show_inherited_class_members = {k: False for k in find_module(plenoptic)}

# to avoid this issue https://stackoverflow.com/a/73294408/4659293
numpydoc_class_members_toctree = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# these are for the sphinx_rtd_theme
html_theme_options = {
    "display_version": True,
}

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
apidoc_module_dir = "../src/plenoptic"

# because of the examples in the docstrings, want to default to showing source and not
# showing link. in actual doc pages, this needs to be reversed.
plot_include_source = True
plot_html_show_source_link = False

# MYST

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "fieldlist",
]

# sphinxcontrib-bibtex
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# sphinx copybutton
copybutton_exclude = ".linenos, .gp"
