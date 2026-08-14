#!/usr/bin/env python3

import pathlib
import sys

import yaml

paths = []
for p in sys.argv[1:]:
    p = pathlib.Path(p)
    if p.is_dir():
        p = list(p.glob("**/*.md"))
    elif p.suffix == ".md":
        p = [p]
    else:
        p = []
    paths.extend(p)


error_files = []
for p in paths:
    txt = p.read_text()
    # then there's no yaml frontmatter
    if not txt.startswith("---"):
        continue
    frontmatter = yaml.safe_load(txt.split("---")[1])
    try:
        format = frontmatter["jupytext"]["text_representation"]["format_name"]
    except KeyError:
        error_files.append(p)
    if format != "myst":
        error_files.append(p)


if error_files:
    print(
        "The following text notebooks are not in Myst format! See the yaml frontmatter "
        "at the top of existing notebooks for examples."
    )
    for f in error_files:
        print(f"\t{f}")
    sys.exit(1)
