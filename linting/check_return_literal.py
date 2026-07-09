#!/usr/bin/env python3
"""
Raise error if we ever have a literal return annotation.

Realized that if a function has a Literal return annotation, e.g.,:

def func() -> Literal["x"]:
   return "x"

then ipython's autocomplete doesn't work; if a single method in a class has this, then
all the autocompletes fail. found this issue about it
(https://github.com/ipython/ipython/issues/14412), which discovered it's a jedi issue
(https://github.com/davidhalter/jedi/issues/1990) which they haven't fixed for over two
years.

to avoid this, just ensure that we don't have any of them (replace with the generic
version instead).

This checks in the simplest possible way: looking for the string "-> Literal".

I also found out that returning a tuple with Literal in it, e.g.,:

def func() -> tuple[Literal["x"], int]:
    return "x", 1

does not have this problem.
"""

import pathlib
import sys

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

error_files = []
for p in paths:
    txt = p.read_text()
    if "-> Literal" in txt:
        error_files.append(p)

if error_files:
    print("return Literal annotation found in following files:")
    for f in error_files:
        print(f"\t{f}")
    sys.exit(1)
