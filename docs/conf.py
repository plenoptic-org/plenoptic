# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import csv
import glob
import inspect
import os
import pathlib
from importlib.metadata import version

import torch
from docutils import nodes

from plenoptic import _api_change

# by default, torch uses all avail threads which slows things run in parallel
torch.set_num_threads(1)


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

if not os.environ.get("SKIP_MPL"):
    extensions.append("matplotlib.sphinxext.plot_directive")
else:
    print(
        "Not running Matplotlib plot directive blocks. This will result "
        "in a lot of warnings."
    )

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

# this is supposed to be true by default
# (https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-the-python-domain)
# but must get set False by one of our extensions
add_module_names = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_favicon = "_static/plenoptic.ico"

html_theme_options = {
    "icon_links": [
        {
            "name": "Home",
            "url": "https://plenoptic.org",
            "icon": "fa-solid fa-house",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/plenoptic-org/plenoptic",
            "icon": "fab fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/plenoptic",
            "icon": "fa-custom fa-pypi",
            "type": "fontawesome",
        },
    ],
    "logo": {
        "image_light": "_static/images/Plenoptic_Logo_CMYK_Full_Wide.svg",
        "image_dark": "_static/images/Plenoptic_Logo_CMYK_Full_DarkMode_Wide.svg",
    },
    "show_prev_next": True,
    "secondary_sidebar_items": {
        # this glob pattern matches anything except a string that starts with api.
        # unfortunately pydata-sphinx raises a warning if a page matches multiple glob
        # patterns (and doesn't assign a reference code so we can tell sphinx to
        # suppress it:
        # https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-suppress_warnings)
        "[!a]?[!p]?[!i]**": ["page-toc"],
        "api/**": [],
    },
    "show_nav_level": 2,
    "header_links_before_dropdown": 4,
    "navbar_align": "left",
    "navbar_start": ["navbar-logo", "version-switcher"],
    "show_version_warning_banner": True,
    "switcher": {
        "json_url": "https://docs.plenoptic.org/docs/branch/main/_static/version_switcher.json",
        "version_match": version,
    },
}

# Path for static files (custom stylesheets or JavaScript)
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = [("custom-icon.js", {"defer": "defer"})]

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

# if release and version match, then we're building docs for a release, else we should
# point to main
tag = release if release == version else "HEAD"

binder_url = f"https://mybinder.org/v2/gh/plenoptic-org/plenoptic-binder/{tag}?urlpath=%2Fdoc%2Ftree%2Fplenoptic%2Fnotebooks%2F"

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
    "linkify",
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
        all_md = pathlib.Path(".").glob("**/*md")
        all_nbs = [
            pathlib.Path(n).stem
            for n in all_md
            if n.read_text().startswith("---\njupytext")
        ]
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

api_order = [
    "synthesis.rst",
    "models.rst",
    "metrics.rst",
    "top_level.rst",
    "display.rst",
    "process.rst",
    "images.rst",
    "validation.rst",
    "optimization.rst",
    "debugging.rst",
    "external.rst",
]
api_dir = pathlib.Path("api")
# API index page (api/index.rst) is auto-generated. It starts with a hidden toctree
# including all the rst pages above, and then includes their text (in order), without
# the sphinx anchor and the toctree argument to the autosummary directive
api_index = """.. _api:

API
===

.. toctree::
   :hidden:

"""
api_index += "   "
api_index += "\n   ".join(mod.replace(".rst", "") for mod in api_order)
api_index += "\n"

for api_rst in api_order:
    api_rst = api_dir / api_rst
    contents = api_rst.read_text().split("\n")
    # two lines we want to throw away: the sphinx anchor (e.g., ".. _synthesis-api") and
    # the line that tells autosummary to create a toctree (e.g., ":toctree: generated")
    contents = [
        c
        for c in contents
        if not c.strip().startswith(".. _") and not c.strip().startswith(":toctree:")
    ]
    api_index += "\n".join(contents)

(api_dir / "index.rst").write_text(api_index)


# Copied and modified from scikit-learn, also suggested by pydata theme
# (https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/static_assets.html#use-an-event-to-add-it-to-specific-pages)
def add_js_css_files(app, pagename, templatename, context, doctree):
    """Load additional JS and CSS files only for certain pages.

    Note that the html_js_files and html_css_files variables are included in all pages
    and should be used for the ones that are used by multiple pages. All page-specific
    JS and CSS files should be added here instead.
    """
    if pagename == "reference/migration_guide":
        # External: DataTables and jQuery
        app.add_js_file("https://code.jquery.com/jquery-3.7.0.js")
        app.add_js_file("https://cdn.datatables.net/2.0.0/js/dataTables.min.js")
        app.add_css_file(
            "https://cdn.datatables.net/2.0.0/css/dataTables.dataTables.min.css"
        )
        # Internal: API search initialization and styling
        app.add_js_file("search-table.js")


# this sphinx event allows us to have fine-grained control over whether to document
# objects or not
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-skip-member
# however, we need an extra step to determine which of *our* objects the
# object-to-document is attached to (https://github.com/sphinx-doc/sphinx/issues/9533)
def skip_torch_inherited_methods(app, obj_type, name, obj, skip, options):
    if obj_type in ("method", "property", "attribute"):
        docobj = None
        for frame in inspect.stack():
            if frame.function == "_get_members":
                docobj = frame.frame.f_locals["obj"]
        if docobj is None:
            raise Exception(
                "Stack of sphinx events has changed, so unsure how to"
                " grab object that corresponds to this method! See "
                "PR #413 for discussion."
            )
        docobj_module = getattr(docobj, "__module__", "")
        # we skip the attributes inherited from torch.nn.Module for our models and model
        # components (found in the process module, we probably never want to show these
        # attributes, but this is a more conservative way of doing this)
        if docobj_module is not None and (
            docobj_module.startswith("plenoptic.models")
            or docobj_module.startswith("plenoptic.process")
        ):
            if obj_type == "method":
                obj_module = getattr(obj, "__module__", "")
                if obj_module is not None and obj_module.startswith("torch.nn.modules"):
                    return True
            else:
                # for some reason, training doesn't show up as inherited (in the
                # following set up or as part of the autodoc's inherited_members that we
                # have access to in the jinja templates), so we exclude it manually
                if name == "training":
                    return True
                # unlike methods, can't just check the module of an attribute, since it
                # will typically be a basic type (e.g., bool). instead, we go through
                # all the classes of docobj and see which one contains the attribute
                # (https://stackoverflow.com/a/42503785/4659293)
                obj_module = None
                for cls in docobj.mro():
                    if name in vars(cls):
                        obj_module = cls.__module__
                if obj_module is not None and obj_module.startswith("torch.nn.modules"):
                    return True
    return None


def binder_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Create a role to highlight new contributors in changelog entries."""
    refuri = binder_url + text
    ref_node = nodes.reference(
        rawtext, text, classes=["binder"], refuri=refuri, **options
    )
    return [ref_node], []


migration_table = pathlib.Path("reference/migration_table.csv")

table = [["plenoptic 1.x", "plenoptic 2.0"]]

UPDATED_API = _api_change.API_CHANGE
UPDATED_API.update(_api_change.SYNTH_PLOT_FUNCS)
UPDATED_API.update(_api_change.PLOT_FUNCS)

for k, v in UPDATED_API.items():
    table.append([f"`{k}`", f"{{func}}`{v}`"])

with migration_table.open("w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(table)

deprecated_table = migration_table.with_stem("deprecated_table")
table = []

for k in _api_change.DEPRECATED:
    table.append([f"`{k}`"])

with deprecated_table.open("w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(table)


# connect our custom method to the sphinx events callback API:
# https://www.sphinx-doc.org/en/master/extdev/event_callbacks.html
def setup(app):
    app.connect("autodoc-skip-member", skip_torch_inherited_methods)
    # triggered just before the HTML for an individual page is created
    app.connect("html-page-context", add_js_css_files)
    app.add_role("binder", binder_role)
