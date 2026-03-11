#!/usr/bin/env python3

import pathlib
import sys

error_files = []
for p in sys.argv[1:]:
    txt = pathlib.Path(p).read_text()
    if "%matplotlib inline" in txt:
        error_files.append(p)

if error_files:
    print("matplotlib inline directive found in following files:")
    for f in error_files:
        print(f"\t{f}")
    sys.exit(1)
