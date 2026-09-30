#!/usr/bin/env python3

import pathlib
import sys

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

mpl_files = []
autoreload_files = []
for p in paths:
    txt = p.read_text()
    if "%matplotlib inline" in txt:
        mpl_files.append(p)
    if "%autoreload" in txt or "%load_ext autoreload" in txt:
        autoreload_files.append(p)

if mpl_files or autoreload_files:
    if mpl_files:
        print("matplotlib inline directive found in following files:")
        for f in mpl_files:
            print(f"\t{f}")
    if autoreload_files:
        print("magic autoreload function found in following files:")
        for f in autoreload_files:
            print(f"\t{f}")
    sys.exit(1)
