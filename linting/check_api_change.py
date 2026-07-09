#!/usr/bin/env python3

# confirm that the _api_change.py file specifies all public functions

import pathlib
import re
import sys

# These are the files whose contents we don't want in the api docs. that's __init__.py
# and those whose contents are meant for internal use
EXCLUDE_MODULES = ["__init__.py", "conftest.py"]

src_modules = pathlib.Path("src/plenoptic").glob("**/*.py")
src_modules = [m for m in src_modules if m.name not in EXCLUDE_MODULES]
# match public functions and classes that aren't nested (i.e., no tabs, don't start with
# underscore)
src_pattern = re.compile(r"^(?:def|class) ([A-Za-z].*)\(", flags=re.MULTILINE)

sys.path.append("src/plenoptic/")
import _api_change  # noqa: E402

UPDATED_API = _api_change.API_CHANGE
UPDATED_API.update(_api_change.SYNTH_PLOT_FUNCS)
UPDATED_API.update(_api_change.PLOT_FUNCS)
NEW_FUNCS = _api_change.NEW

ALL_FUNCS = set(UPDATED_API.values()).union(NEW_FUNCS + _api_change.UNCHANGED)
# making this behave similar to the dict in check_apidocs.py
match_dict = {}
for func in ALL_FUNCS:
    func_split = func.split(".")
    mod = ".".join(func_split[:-1])
    if mod in match_dict:
        match_dict[mod].append(func_split[-1])
    else:
        match_dict[mod] = [func_split[-1]]


src_not_api_change = []
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
        src_not_api_change.append((module, missing_objs))


if src_not_api_change:
    print(
        "The following public functions/classes are not found in the "
        "_api_change.py file!"
    )
    for mod, objs in src_not_api_change:
        print(f"{mod}:")
        for ob in objs:
            print(f"\t{ob}")
    sys.exit(1)
