#!/usr/bin/env python3

import ast
import pathlib
import re
import sys

LINK_REGEX = re.escape("](")
# we want basically everything except a backtick
EVERYTHING_BUT_BACKTICK = r"[A-Za-z_\(\)<>:/ -.=\[\]0-9\\^~{}|'\"]"
BACKTICK_REGEX = rf"(`+{EVERYTHING_BUT_BACKTICK}+?`+_?)"
SPHINX_DIRECTIVE_REGEX = rf":[:a-z]+:(`+{EVERYTHING_BUT_BACKTICK}+?_?`+)"
SPHINX_LINK_REGEX = rf"(`+{EVERYTHING_BUT_BACKTICK}+?`+_)"


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
backticks = []

for p in paths:
    with open(p) as f:
        tree = ast.parse(f.read())
    docstrings = get_docstrings(tree.body)

    for name, doc in docstrings:
        if re.findall(LINK_REGEX, doc):
            links.append((p, name))
        directives = re.findall(SPHINX_DIRECTIVE_REGEX, doc)
        sphinx_link = re.findall(SPHINX_LINK_REGEX, doc)
        backtick = [
            x
            for x in re.findall(BACKTICK_REGEX, doc)
            if x not in directives + sphinx_link and not x.startswith("``")
        ]
        if backtick:
            backticks.append((p, name, backtick))

if backticks or links:
    if backticks:
        print("The following docstrings appear to contain markdown:")
        for p, name, markup in backticks:
            markup = ", ".join(markup)
            print(f"{p}:{name} {markup}")
    if links:
        print("The following docstrings appear to contain markdown links:")
        for p, name in links:
            print(f"{p}:{name}")
    sys.exit(1)
