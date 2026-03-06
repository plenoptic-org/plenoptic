#!/usr/bin/env python3

import pathlib
import re
import subprocess
import sys

# These are the files whose contents we don't want in the api docs. that's __init__.py
# and those whose contents are meant for internal use
EXCLUDE_MODULES = ["__init__.py", "autodiff.py", "synthesis.py", "classes.py"]

src_modules = pathlib.Path("src/plenoptic").glob("**/*.py")
src_modules = [m for m in src_modules if m.name not in EXCLUDE_MODULES]
# match public functions and classes that aren't nested (i.e., no tabs, don't start with
# underscore)
src_pattern = re.compile(r"^(?:def|class) ([A-Za-z].*)\(", flags=re.MULTILINE)

# generate all the stub files
rst_files = [str(f) for f in pathlib.Path("docs/api").glob("*rst")]
subprocess.run(["sphinx-autogen", *rst_files])

match_dict = {}
stub_files = pathlib.Path("docs/api/generated").glob("*.rst")
for f in stub_files:
    name = f.stem.split(".")
    mod = ".".join(name[:-1])
    obj = name[-1]
    if mod not in match_dict:
        match_dict[mod] = [obj]
    else:
        match_dict[mod].append(obj)

src_not_api = []
for module in src_modules:
    api_objs = []
    # last two here will be . and ./src, which we can drop
    possible_modules = [f.name for f in module.parents][:-2]
    # reverse this so it goes from biggest (plenoptic) to smallest in scope
    possible_modules = possible_modules[::-1]
    # add the current module at the end
    possible_modules.append(module.stem)
    mod_name = ""
    for p in possible_modules:
        mod_name = ".".join([mod_name, p])
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
