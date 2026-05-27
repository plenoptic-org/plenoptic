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

error_files = []
for p in paths:
    txt = p.read_text()
    if "%matplotlib inline" in txt:
        error_files.append(p)

if error_files:
    print("matplotlib inline directive found in following files:")
    for f in error_files:
        print(f"\t{f}")
    sys.exit(1)
