#!/usr/bin/env python

import ast
import pathlib
import re
import sys

with open(sys.argv[1]) as f:
    tests = ast.parse(f.read())

tests = [
    c
    for c in tests.body
    if isinstance(c, ast.ClassDef) and c.name == "TestTutorialNotebooks"
]
assert len(tests) == 1
tests = {
    test_class.name: {
        test_func.name: ast.unparse(test_func.body)
        for test_func in test_class.body
        if test_func.name.startswith("test")
    }
    for test_class in tests[0].body
}

paths = []
for p in sys.argv[2:]:
    p = pathlib.Path(p)
    if p.is_dir():
        p = list(p.glob("**/*.md"))
    elif p.suffix == ".md":
        p = [p]
    else:
        p = []
    paths.extend(p)

nb_re_str = r"<!-- ?(Test[A-z_\.]+)(\[[a-z:,_ ]+\])? ?-->\n```.*?\n(.*?)```"

match_not_found = []
test_mismatch = []
for p in paths:
    with open(p) as f:
        md = f.read()
    if not md.startswith("---\njupytext"):
        # then this isn't a markdown notebook
        continue
    synth_check_blocks = re.findall(nb_re_str, md, flags=re.DOTALL)
    if not synth_check_blocks:
        # then there's nothing to check
        continue
    for test_name, nb_replace, nb_contents in synth_check_blocks:
        test_class, test_func = test_name.split(".")
        try:
            func_body = tests[test_class][test_func]
        except KeyError:
            match_not_found.append((p, test_name))
            continue
        if nb_replace:
            for rep in nb_replace[1:-1].split(","):
                nb_contents = nb_contents.replace(*rep.split(":"))
        func_body = "\n".join(
            [line for line in func_body.splitlines() if "lint_ignore" not in line]
        )
        nb_contents = ast.unparse(ast.parse(nb_contents))
        if nb_contents not in func_body:
            test_mismatch.append((p, test_name, nb_contents))


if match_not_found or test_mismatch:
    if match_not_found:
        print(
            "Each synthesize block in a notebook needs a corresponding test, the "
            "following failed:"
        )
        for p, missing in match_not_found:
            print(f"{p}: {missing=}")
    if test_mismatch:
        print("Test found but did not match for the following:")
        for p, test_name, nb_contents in test_mismatch:
            print(f"{p}: {test_name=}\n\n{nb_contents}\n")
    sys.exit(1)
