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

test_re_str = r"(?:\n.*setup\((.*)\)\n)?.*synthesize\((.*)\)"
nb_re_str = rf"```.*\n:name: *(test.*)\n*{test_re_str}\n*```"
match_not_found = []
synth_wrong = []
setup_wrong = []
for p in paths:
    with open(p) as f:
        md = f.read()
    if not md.startswith("---\njupytext"):
        # then this isn't a markdown notebook
        continue
    synth_check_blocks = re.findall(nb_re_str, md)
    check_funcs = tests.get(f"Test{p.stem.replace('_', '')}", None)
    if check_funcs is None:
        check_funcs = {}
    else:
        check_funcs = {
            b.name: ast.unparse(b.body)
            for b in check_funcs.body
            if b.name.startswith("test")
        }
    checked = [b[0] not in check_funcs for b in synth_check_blocks]
    if any(checked):
        fail_blocks = [b[0] for b, t in zip(synth_check_blocks, checked) if t]
        match_not_found.append((p, fail_blocks))
        continue
    for test_name, setup_args, synth_args in synth_check_blocks:
        func_body = check_funcs[test_name]
        test_setup_args, test_synth_args = re.findall(test_re_str, func_body)[0]
        # normalize the quotes
        test_synth_args = test_synth_args.replace("'", '"')
        synth_args = synth_args.replace("'", '"')
        if test_synth_args != synth_args:
            synth_wrong.append((p, test_name, test_synth_args, synth_args))
        # normalize the quotes
        test_setup_args = test_setup_args.replace("'", '"')
        setup_args = setup_args.replace("'", '"')
        if test_setup_args != setup_args:
            setup_wrong.append((p, test_name, test_setup_args, setup_args))


if match_not_found or synth_wrong or setup_wrong:
    if match_not_found:
        print(
            "Each synthesize block in a notebook needs a corresponding test, the "
            "following failed:"
        )
        for p, missing in match_not_found:
            print(f"{p}: {missing=}")
    if synth_wrong:
        print("Synthesize args did not match for the following:")
        for p, func_name, test_args, nb_args in synth_wrong:
            print(f"{p}: {func_name=}\n\t{test_args} (test)\n\t{nb_args} (notebook)")
    if setup_wrong:
        print("Setup args did not match for the following:")
        for p, func_name, test_args, nb_args in setup_wrong:
            print(f"{p}: {func_name=}\n\t{test_args} (test)\n\t{nb_args} (notebook)")
    sys.exit(1)
