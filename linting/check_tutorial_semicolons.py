#!/usr/bin/env python3

import os
import pathlib
import subprocess
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


os.makedirs("nb_tmp", exist_ok=True)
anim_errors = {}
plot_errors = {}
for p in paths:
    with open(p) as f:
        md = f.read()
    if not md.startswith("---\njupytext"):
        # then this isn't a markdown notebook
        continue
    script_path = f"nb_tmp{os.sep}{p.stem}.py"
    subprocess.run(
        ["jupytext", p, "-o", script_path, "--from", "myst"], capture_output=True
    )
    with open(script_path) as f:
        txt = f.readlines()
    open_parens = 0
    close_parens = 0
    plot_func = False
    anim_func = False
    # the conversion to .py from .md adds `# %% [markdown]` lines at the beginning of
    # every markdown block, so we keep track of them here to offset them in the error
    # message. start with 2, because the conversion adds "jupyter:" and "default_lexer:"
    # lines
    script_offset = 2
    for i, line in enumerate(txt):
        # remove ending whitespace
        line = line.strip()
        # we ignore comments and markdown
        if line.startswith("#"):
            continue
        if "po.plot" in line:
            # reset parens counts
            open_parens = 0
            close_parens = 0
            if "animshow" in line:
                anim_func = True
            else:
                plot_func = True
            func_line = line
        open_parens += line.count("(")
        close_parens += line.count(")")
        if close_parens > open_parens:
            raise ValueError("Something weird with my parens counting")
        elif open_parens == close_parens and (anim_func or plot_func):
            # then we've closed the function call
            if anim_func:
                if line.endswith(";"):
                    # the +1 is because the enumerate iteration starts at 0, but lines
                    # start at 1
                    if p not in anim_errors:
                        anim_errors[p] = [(func_line, i - script_offset + 1)]
                    else:
                        anim_errors[p].append((func_line, i - script_offset + 1))
                anim_func = False
            if plot_func:
                if not line.endswith(";"):
                    # the +1 is because the enumerate iteration starts at 0, but lines
                    # start at 1
                    if p not in plot_errors:
                        plot_errors[p] = [(func_line, i - script_offset + 1)]
                    else:
                        plot_errors[p].append((func_line, i - script_offset + 1))
                plot_func = False


if anim_errors:
    print("animate functions found ending in semicolon -- remove them:\n")
    for f, errors in anim_errors.items():
        print(f)
        for line, idx in errors:
            print(f"\tline ~{idx}: {line}")
        print()

if plot_errors:
    print("plot functions found not ending in semicolon -- add one:\n")
    for f, errors in plot_errors.items():
        print(f)
        for line, idx in errors:
            print(f"\tline ~{idx}: {line}")
        print()


if anim_errors or plot_errors:
    sys.exit(1)
