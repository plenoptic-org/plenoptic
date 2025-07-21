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
tests = {b.name: b for b in tests[0].body}


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

match_not_found = []
wrong_match = []
for p in paths:
    with open(p) as f:
        md = f.read()
    if not md.startswith("---\njupytext"):
        # then this isn't a markdown notebook
        continue
    # block should only contains the name (on the line right after the block is
    # opened) and the synthesize line (with optional blank lines)
    synth_check_blocks = re.findall(
        "```.*\n:name: *(test.*)\n*.*synthesize\\((.*\n?.*)\\)\n*```", md
    )
    check_funcs = tests.get(f"Test{p.stem.replace('_', '')}", None)
    if check_funcs is None:
        check_funcs = {}
    else:
        check_funcs = {b.name: ast.unparse(b.body) for b in check_funcs.body}
    if len(synth_check_blocks) != len(check_funcs):
        match_not_found.append((p, len(synth_check_blocks), len(check_funcs)))
        continue
    for test_name, synth_args in synth_check_blocks:
        func_body = check_funcs[test_name]
        test_args = re.findall(r".*synthesize\((.*)\)", func_body)[0]
        # normalize the quotes
        test_args = test_args.replace("'", '"')
        synth_args = synth_args.replace("'", '"')
        if test_args != synth_args:
            wrong_match.append((p, test_name, test_args, synth_args))


if match_not_found or wrong_match:
    if match_not_found:
        print(
            "Didn't find the proper number of synthesize calls in the following"
            " notebooks:"
        )
        for p, found, target in match_not_found:
            print(f"{p}: {found=}, {target=}")
    if wrong_match:
        print("Synthesize args did not match for the following:")
        for p, func_name, test_args, nb_args in wrong_match:
            print(f"{p}: {func_name=}\n\t{test_args} (test)\n\t{nb_args} (notebook)")
    sys.exit(1)
