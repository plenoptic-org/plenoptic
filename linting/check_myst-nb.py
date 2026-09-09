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
kernel_name_error = []
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
    else:
        if format != "myst":
            error_files.append(p)
    try:
        kernel_name = frontmatter["kernelspec"]["name"]
    except KeyError:
        error_files.append(p)
    else:
        if kernel_name != "python3":
            kernel_name_error.append(p)


if error_files or kernel_name_error:
    if error_files:
        print(
            "The following text notebooks are not in Myst format (or their "
            "frontmatter is mis-formatted)! See the yaml frontmatter at the "
            "top of existing notebooks for examples."
        )
        for f in set(error_files):
            print(f"\t{f}")
    if kernel_name_error:
        print(
            "The following text notebooks have a non-standard kernel name! Set "
            "kernelspec:name in the yaml frontmatter to 'python3'. See existing "
            "notebooks for examples."
        )
        for f in kernel_name_error:
            print(f"\t{f}")
    sys.exit(1)
