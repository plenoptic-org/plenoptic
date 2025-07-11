#!/usr/bin/env python3

import ast
import pathlib
import re
import sys

LINK_REGEX = re.escape("](")
# we want basically everything except a backtick
EVERYTHING_BUT_BACKTICK = r"[ A-Za-z_\(\)<>:/\-.=\[\]0-9\\^~{}|'\",]"
BACKTICK_REGEX = rf"(`+{EVERYTHING_BUT_BACKTICK.replace(' ', '')}+?`+_?)"
SPHINX_DIRECTIVE_REGEX = rf":[:a-z]+:(`+{EVERYTHING_BUT_BACKTICK}+?_?`+)"
SPHINX_LINK_REGEX = rf"(`+{EVERYTHING_BUT_BACKTICK}+?`+_)"
# underscores are part of variable names, so we don't want to match them
EVERYTHING_BUT_BACKTICK_ = EVERYTHING_BUT_BACKTICK.replace("_", "")
# None/True/False surrounded by single backtick is also wrong, but will be caught by the
# markdown check.
COMMON_VALS = (
    rf"{EVERYTHING_BUT_BACKTICK_}([Nn]one|[Tt]rue|[Ff]alse){EVERYTHING_BUT_BACKTICK_}"
)
LIBRARIES_XREF = [
    "plenoptic",
    "po",
    "torch",
    "numpy",
    "np",
    "matplotlib",
    "mpl",
    "plt",
    "pyrtools",
    "pt",
]
# we want to grab the references to above libraries that should be xrefs. So we don't
# want to match: proper xrefs (which will have a ~ or ` before the library name) or urls
# (which will have . or / before the nmae)
XREF_EXCEPT = EVERYTHING_BUT_BACKTICK.replace("~", "").replace(".", "").replace("/", "")
# additionally, = and ( show up in code snippets
XREF_EXCEPT = XREF_EXCEPT.replace("=", "").replace("(", "")


def parse_docstring(docstring):
    docstring = docstring.splitlines()
    docstring = [d.strip() for d in docstring]
    return " ".join(docstring)


def get_docstrings(tree_body):
    docstrings = []
    for node in tree_body:
        try:
            doc = ast.get_docstring(node)
        except TypeError:
            doc = None
        if doc is not None:
            docstrings.append((node.name, parse_docstring(doc)))
    return docstrings


paths = []
for p in sys.argv[1:]:
    p = pathlib.Path(p)
    if p.is_dir():
        p = list(p.glob("**/*.py"))
    elif p.suffix == ".py":
        p = [p]
    else:
        p = []
    paths.extend(p)

links = []
defaults = []
backticks = []
unescaped = []
missing_xref = []
dollarsigns = []

for p in paths:
    with open(p) as f:
        tree = ast.parse(f.read())
    docstrings = get_docstrings(tree.body)

    for name, doc in docstrings:
        if re.findall(LINK_REGEX, doc):
            links.append((p, name))
        if re.findall(r"\$", doc):
            dollarsigns.append((p, name))
        directives = re.findall(SPHINX_DIRECTIVE_REGEX, doc)
        sphinx_link = re.findall(SPHINX_LINK_REGEX, doc)
        # if the word default is preceded / followed by an underscore, then the
        # argument has "default" in its name, so don't match that (using
        # negative look-ahead/behind)
        default = re.findall("(?<!_)[Dd]efault(?!_)", doc)
        if default:
            defaults.append((p, name))
        backtick = [
            x
            for x in re.findall(BACKTICK_REGEX, doc)
            if x not in directives + sphinx_link and not x.startswith("``")
        ]
        if backtick:
            backticks.append((p, name, backtick))
        # See Also and Examples (the last two sections, fortunately) can both contain
        # un-xref'ed references to other libraries or None/False/True
        doc = doc.split("See Also")[0].split("Examples")[0]
        if unesc := re.findall(COMMON_VALS, doc):
            unescaped.append((p, name, unesc))
        for lib in LIBRARIES_XREF:
            # use negative look-behind to avoid stuff in the attribute section
            if xr := re.findall(rf"(?<![a-z]:){XREF_EXCEPT}{lib}\.", doc):
                # the url for torch has `pytorch.` (not `torch.`), which we don't want
                # to match
                if lib == "torch":
                    xr = [x for x in xr if x != "ytorch."]
                if not xr:
                    continue
                missing_xref.append((p, name, xr))
            # we also want to raise an error if we have two backticks before these
            # libraries, which corresponds to monospace (but no link). We *don't* want
            # to match if it looks like an actual call, so we either can have no
            # parentheses or only parentheses with on args in them at the end of the
            # backticks
            if xr := re.findall(rf"``{lib}\.[a-z0-9_\.]+(?:\(\))?``", doc):
                missing_xref.append((p, name, xr))

if backticks or links or unescaped or missing_xref or dollarsigns or defaults:
    if backticks:
        print("The following docstrings appear to contain markdown:")
        for p, name, markup in backticks:
            markup = ", ".join(markup)
            print(f"{p}:{name} {markup}")
        print("\n")
    if links:
        print("The following docstrings appear to contain markdown links:")
        for p, name in links:
            print(f"{p}:{name}")
        print("\n")
    if unescaped:
        print(
            "The following docstrings appear to contain values that should be"
            " surounded in backticks:"
        )
        for p, name, markup in unescaped:
            print(f"{p}:{name} {markup}")
        print("\n")
    if missing_xref:
        print(
            "The following docstrings appear to contain references to other functions;"
            "should use xref:"
        )
        for p, name, markup in missing_xref:
            print(f"{p}:{name} {markup}")
        print("\n")
    if dollarsigns:
        print(
            "The following docstrings appear to contain latex surrounded by dollar "
            "signs; docstrings should be rst-formatted, use :math:`x` instead of $x$"
        )
        for p, name in dollarsigns:
            print(f"{p}:{name}")
    if defaults:
        print(
            "The following docstrings appear to contain a description of a default "
            "argument. Remove that, it's hard to keep up-to-date."
        )
        for p, name in defaults:
            print(f"{p}:{name}")
    sys.exit(1)
