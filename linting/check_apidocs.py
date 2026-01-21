#!/usr/bin/env python3

import pathlib
import re
import sys

# These are the files whose contents we don't want in the api docs. that's __init__.py
# and those whose contents are meant for internal use
EXCLUDE_MODULES = ["__init__.py", "autodiff.py", "synthesis.py", "classes.py"]

src_modules = pathlib.Path("src/plenoptic").glob("**/*.py")
src_modules = [m for m in src_modules if m.name not in EXCLUDE_MODULES]
# match public functions and classes that aren't nested (i.e., no tabs, don't start with
# underscore)
src_pattern = re.compile(r"^(?:def|class) ([A-Za-z].*)\(", flags=re.MULTILINE)

api_doc = pathlib.Path("docs/api.rst").read_text()

# match all strings that are: some number of spaces, then a tilde, then a string with at
# least one dot in the middle of it. the goal is to match lines like
# `~metamer.plot_loss`, `~synthesize.metamer.plot_loss` (but not `Metamer`)
api_pattern = re.compile(r"^ +~(.+\.*)\.(.+)", flags=re.MULTILINE)
matches = re.findall(api_pattern, api_doc)
match_dict = {}
for mod, obj in matches:
    if mod not in match_dict:
        match_dict[mod] = [obj]
    else:
        match_dict[mod].append(obj)

src_not_api = []
for module in src_modules:
    api_objs = []
    parents = [""] + [f.name for f in module.parents]
    mod_name = module.stem
    for p in parents:
        mod_name = ".".join([p, mod_name])
        if mod_name[0] == ".":
            mod_name = mod_name[1:]
        api_objs.extend(match_dict.get(mod_name, []))
    module_text = module.read_text()
    src_objs = re.findall(src_pattern, module_text)
    if missing_objs := set(src_objs) - set(api_objs):
        src_not_api.append((module, missing_objs))


if src_not_api:
    print("The following public functions/classes are not found in the API docs!")
    for mod, objs in src_not_api:
        print(f"{mod}:")
        for ob in objs:
            print(f"\t{ob}")
    sys.exit(1)
