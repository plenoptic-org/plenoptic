#!/usr/bin/env python3

import pathlib
import re
import sys

import astroid

LINK_REGEX = re.escape("](")
# we want basically everything except a backtick
EVERYTHING_BUT_BACKTICK = r"[A-Za-z_\(\)<>:/ -.=\[\]0-9\\^~{}'\"]"
BACKTICK_REGEX = rf"(`+{EVERYTHING_BUT_BACKTICK}+?`+_?)"
SPHINX_DIRECTIVE_REGEX = rf":[:a-z]+:(`+{EVERYTHING_BUT_BACKTICK}+?_?`+)"
SPHINX_LINK_REGEX = rf"(`+{EVERYTHING_BUT_BACKTICK}+?`+_)"


def parse_docstring(doc_node):
    if doc_node is not None and doc_node.value is not None:
        docstring = doc_node.value.splitlines()
        docstring = [d.strip() for d in docstring]
        return " ".join(docstring)
    else:
        return ""


def get_nodes(tree_body):
    funcs = []
    classes = []
    for obj in tree_body:
        try:
            obj_type = obj.type
        except AttributeError:
            obj_type = ""
        if obj_type in ["function", "method"]:
            funcs.append(obj)
        elif obj_type == "class":
            classes.append(obj)
            f, c = get_nodes(obj.body)
            funcs.extend(f)
            classes.extend(c)
    return funcs, classes


paths = []
for p in sys.argv[1:]:
    p = pathlib.Path(p)
    p = list(p.glob("**/*.py")) if p.is_dir() else [p]
    paths.extend(p)

links = []
backticks = []

for p in paths:
    with open(p) as f:
        tree = astroid.parse(f.read())
    funcs, classes = get_nodes(tree.body)
    docstrings = [(c.name, parse_docstring(c.doc_node)) for c in classes] + [
        (f.name, parse_docstring(f.doc_node)) for f in funcs
    ]

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
