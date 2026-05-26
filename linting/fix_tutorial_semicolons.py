#!/usr/bin/env python3

import json
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
updated_any_file = False
for p in paths:
    with open(p) as f:
        md = f.read()
    if not md.startswith("---\njupytext"):
        # then this isn't a markdown notebook
        continue
    nb_path = f"nb_tmp{os.sep}{p.stem}.ipynb"
    subprocess.run(
        ["jupytext", p, "-o", nb_path, "--from", "myst"], capture_output=True
    )
    with open(nb_path) as f:
        nb_txt = json.load(f)
    updated_file = False
    for i, cell in enumerate(nb_txt["cells"]):
        # only look at code cells
        if cell["cell_type"] != "code":
            continue
        # we are looking for the last python expression in each cell. because we run
        # ruff format on our notebooks, we can assume they're properly indented.
        #
        # First, grab the non-empty lines
        src = [line for line in cell["source"] if line.strip()]
        # then, grab those lines that don't start with white space
        src = [line for line in src if line.strip()[0] == line[0]]
        # then remove any lines that are just closing parentheses
        src = [line for line in src if not line.startswith(")")]
        # if there's nothing left here, then skip
        if not src:
            continue
        # then grab the beginning of the last expression
        last_expr = cell["source"].index(src[-1])
        # and then the whole thing
        last_expr = cell["source"][last_expr:]
        last_expr = " ".join([line.strip() for line in last_expr])
        # then we've found one of our plotting functions
        if "po.plot" in last_expr and "=" not in last_expr.split("po.plot")[0]:
            # remove the semicolon from animshow functions
            if "animshow" in last_expr and last_expr.endswith(";"):
                cell["source"][-1] = cell["source"][-1][:-1]
                updated_file = True
                updated_any_file = True
            # add semicolon from plotting functions
            elif "animshow" not in last_expr and not last_expr.endswith(";"):
                cell["source"][-1] += ";"
                updated_file = True
                updated_any_file = True
    if updated_file:
        print(f"Updating {p}")
        with open(nb_path, "w") as f:
            json.dump(nb_txt, f)
        subprocess.run(
            ["jupytext", nb_path, "-o", p, "--to", "myst"], capture_output=True
        )


if updated_any_file:
    sys.exit(1)
