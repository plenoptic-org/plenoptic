#!/usr/bin/env python3

import pathlib
import re
import sys

failures = []

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

for p in paths:
    with open(p) as f:
        contents = f.read()
    fail = re.findall(r"\.\.[a-zA-Z]+::", contents)
    if len(fail):
        failures.append((p, fail))

if len(failures):
    print("Found the following malformed sphinx directives! Add a space after the dots")
    for p, fail in failures:
        fail = ", ".join(fail)
        print(f"{p}: {fail}")
    sys.exit(1)
